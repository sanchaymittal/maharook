#!/usr/bin/env python3
"""
Maharook — ROOK fine-tuning (LoRA) on CPU

- Loads core_features.csv + core_conditions.csv
- Generates heuristic labels (bootstrap teacher)
- Builds tiny instruction datasets (prompt → JSON action)
- Runs LoRA fine-tune of a parent model (FinR1 or DeepSeek-R1-Distill)
- Saves adapter to models/adapters/<PAIR>/

Tested for CPU-only, low-RAM. Keep sequence short and batch size = 1.
"""

import os
import json
import math
import argparse
import warnings
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_csvs(features_path: str, conditions_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_feat = pd.read_csv(features_path)
    df_cond = pd.read_csv(conditions_path)
    # Ensure timestamp is comparable
    if not np.issubdtype(df_feat["timestamp"].dtype, np.datetime64):
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])
    if not np.issubdtype(df_cond["timestamp"].dtype, np.datetime64):
        df_cond["timestamp"] = pd.to_datetime(df_cond["timestamp"])
    return df_feat.sort_values("timestamp"), df_cond.sort_values("timestamp")

def join_on_time(df_feat: pd.DataFrame, df_cond: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join conditional rows to features by nearest previous timestamp (asof).
    If conditions has only 1 row, broadcast it to all feature rows.
    """
    df_feat = df_feat.copy()
    df_cond = df_cond.copy()

    # If conditions has only 1 row, broadcast to all features
    if len(df_cond) == 1:
        print(f"→ Broadcasting single condition row to all {len(df_feat)} feature rows")
        # Remove timestamp from conditions and broadcast
        cond_cols = [c for c in df_cond.columns if c != "timestamp"]
        for col in cond_cols:
            df_feat[col] = df_cond[col].iloc[0]
        return df_feat

    # Normal time-based join
    df_feat.set_index("timestamp", inplace=True)
    df_cond.set_index("timestamp", inplace=True)
    joined = pd.merge_asof(df_feat.sort_index(), df_cond.sort_index(), left_index=True, right_index=True, direction="backward")
    joined = joined.dropna().reset_index().rename(columns={"index":"timestamp"})
    return joined

# ----------------------------
# Heuristic Labeling
# ----------------------------

def zscore(series: pd.Series, win: int = 50) -> pd.Series:
    rolling_mean = series.rolling(win, min_periods=max(5, win//5)).mean()
    rolling_std  = series.rolling(win, min_periods=max(5, win//5)).std().replace(0, np.nan)
    return (series - rolling_mean) / rolling_std

def label_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bootstrap labels:
    - side: BUY if z < -k, SELL if z > +k, otherwise HOLD
    - size: proportional to |z|, clipped
    - slippage/deadline: simple mappings from volatility/time_diff
    """
    out = df.copy()
    out["price"] = out["price"].astype(float)
    out["volatility"] = out["volatility"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["time_diff"] = out["time_diff"].astype(float).fillna(0.0)

    z = zscore(out["price"], win=50).fillna(0.0)
    k = 1.0  # threshold

    side = np.where(z > +k, "SELL", np.where(z < -k, "BUY", "HOLD"))
    mag  = np.clip(np.abs(z) / 3.0, 0.05, 1.0)  # 5%..100%

    # map volatility (0..percentile) -> slippage bps (10..50)
    vol = out["volatility"].clip(lower=0.0)
    vq  = np.percentile(vol[vol.notna()], 90) if (vol.notna().sum() > 10) else (vol.max() if vol.max() > 0 else 1.0)
    vol_norm = np.clip(vol / (vq if vq > 0 else 1.0), 0.0, 1.0)
    slip_bps = (10 + 40*vol_norm).astype(float)

    # map time_diff (sec) -> deadline (15..60s)
    td = out["time_diff"].fillna(0.0).astype(float)
    td_norm = np.clip(td / np.percentile(td, 90) if np.percentile(td, 90) > 0 else 1.0, 0.0, 1.0)
    deadline = (15 + 45*td_norm).astype(float)

    out["label_side"] = side
    out["label_size"] = mag.round(3)
    out["label_slippage_bps"] = np.round(slip_bps, 1)
    out["label_deadline_s"] = np.round(deadline, 0)

    return out

# ----------------------------
# Prompt Serialization
# ----------------------------

INSTRUCTION_TMPL = """You are ROOK, a trading agent specializing in {pair}. 
Given recent core and conditional context, output a single JSON object with keys: side, size, slippage_bps, deadline_s.

Constraints:
- side ∈ ["BUY","SELL","HOLD"]
- size ∈ (0,1]
- slippage_bps ∈ [5, 100]
- deadline_s ∈ [10, 120]

Core (latest row):
{core_json}

Conditional (aggregated):
{cond_json}

Respond with ONLY one JSON object, nothing else."""

def row_to_core_json(row: pd.Series, core_cols: List[str]) -> str:
    core = {k: (None if pd.isna(row[k]) else float(row[k])) for k in core_cols if k in row}
    return json.dumps(core, separators=(",", ":"))

def row_to_cond_json(row: pd.Series, cond_cols: List[str]) -> str:
    cond = {k: (None if pd.isna(row[k]) else float(row[k])) for k in cond_cols if k in row}
    return json.dumps(cond, separators=(",", ":"))

def build_prompt(row: pd.Series, pair: str, core_cols: List[str], cond_cols: List[str]) -> str:
    return INSTRUCTION_TMPL.format(
        pair=pair,
        core_json=row_to_core_json(row, core_cols),
        cond_json=row_to_cond_json(row, cond_cols),
    )

def build_label_json(row: pd.Series) -> str:
    y = {
        "side": str(row["label_side"]),
        "size": float(row["label_size"]),
        "slippage_bps": float(row["label_slippage_bps"]),
        "deadline_s": float(row["label_deadline_s"]),
    }
    return json.dumps(y, separators=(",", ":"))

# ----------------------------
# HF Dataset wrapper (minimal)
# ----------------------------

class InstrDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pair: str, core_cols: List[str], cond_cols: List[str], tokenizer, max_len: int = 384):
        self.df = df.reset_index(drop=True)
        self.pair = pair
        self.core_cols = core_cols
        self.cond_cols = cond_cols
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Prebuild texts to avoid prompt regen in __getitem__
        self.sources = []
        self.targets = []
        for _, r in self.df.iterrows():
            prompt = build_prompt(r, pair, core_cols, cond_cols)
            label  = build_label_json(r)
            # Supervised format: "[INST] prompt [/INST] label"
            text = f"<s>[INST] {prompt} [/INST] {label}</s>"
            self.sources.append(text)
            self.targets.append(label)  # not used directly (we train on full text)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        text = self.sources[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        # Labels are the same as input ids (causal LM); mask out prompt if desired (kept simple here)
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}

# ----------------------------
# Training
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="ETH/USDC", help="Pair name for prompts and output dir")
    parser.add_argument("--features", default="core_features.csv", help="Path to core features CSV")
    parser.add_argument("--conditions", default="core_conditions.csv", help="Path to conditional features CSV")
    parser.add_argument("--model_name", default="Mungert/Fin-R1-GGUF", help="Parent model on HF (change if needed)")
    parser.add_argument("--output_dir", default="models/adapters/ETH_USDC", help="Where to save LoRA adapter")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train fraction (time split)")
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("→ Loading CSVs…")
    df_feat, df_cond = load_csvs(args.features, args.conditions)
    df = join_on_time(df_feat, df_cond)

    # Select a conservative set of columns that exist in your CSVs
    core_cols = [
        "price","amount0_eth","amount1_usdc","volume","volatility",
        "volume_ma","tick_change","time_diff","liquidity_impact","price_change","liquidity","tick"
    ]
    cond_cols = [c for c in df.columns if c.startswith(("avg_", "price_volatility", "total_transactions", "price_trend", "volume_trend", "current_"))]

    print("→ Labeling with heuristics…")
    df_labeled = label_actions(df)

    # Time-based split
    n = len(df_labeled)
    split_idx = int(args.train_frac * n)
    df_train = df_labeled.iloc[:split_idx].copy()
    df_val   = df_labeled.iloc[split_idx:].copy()

    print(f"Dataset sizes: train={len(df_train)}, val={len(df_val)}")

    print(f"→ Loading tokenizer & model: {args.model_name}")
    # NOTE: If Fin-R1 repo doesn’t expose a HF Transformers model, switch to a small open model (e.g. Qwen2.5-1.5B)
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,   # CPU
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": "cpu"}
    )

    # Prepare LoRA
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None,  # let PEFT auto-find common projection layers
    )

    # (Optional) if model was 4/8-bit; on CPU we keep float32
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    model = get_peft_model(model, lora_cfg)

    print("→ Building datasets…")
    train_ds = InstrDataset(df_train, args.pair, core_cols, cond_cols, tokenizer, max_len=args.max_len)
    val_ds   = InstrDataset(df_val,   args.pair, core_cols, cond_cols, tokenizer, max_len=args.max_len)

    ensure_dir(args.output_dir)

    # Keep settings *very* light for CPU
    targs = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.0,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        logging_steps=25,
        save_total_limit=2,
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,
        report_to=[],  # disable wandb etc.
        prediction_loss_only=True,
    )

    def data_collator(features):
        batch = {}
        keys = features[0].keys()
        for k in keys:
            batch[k] = torch.stack([f[k] for f in features])
        return batch

    print("→ Starting training (CPU) …")
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter (PEFT)
    print("→ Saving LoRA adapter …")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Minimal manifest
    manifest = {
        "pair": args.pair,
        "parent_model": args.model_name,
        "columns_core": core_cols,
        "columns_cond": list(cond_cols),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Done. Adapter saved to: {args.output_dir}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
