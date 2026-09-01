#!/usr/bin/env python3
"""
DyGIE-- Standalone — 学習スクリプト

使い方:
  python scripts/train.py \
      --config configs/scierc.json \
      --train_path data/train.jsonl \
      --dev_path   data/dev.jsonl \
      --output_dir output/scierc_bert
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from dygie.data import DyGIEDataset, collate_fn
from dygie.model import DyGIE
from dygie.training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DyGIE++ Standalone")
    p.add_argument("--config",      required=True,  help="JSON 設定ファイルのパス")
    p.add_argument("--train_path",  required=True,  help="学習データ (.jsonl)")
    p.add_argument("--dev_path",    required=True,  help="検証データ (.jsonl)")
    p.add_argument("--output_dir",  required=True,  help="出力ディレクトリ")
    # 設定ファイルを CLI で上書きできる主要オプション
    p.add_argument("--transformer_model", default=None)
    p.add_argument("--num_epochs",        type=int,   default=None)
    p.add_argument("--batch_size",        type=int,   default=None)
    p.add_argument("--lr_transformer",    type=float, default=None)
    p.add_argument("--lr_task",           type=float, default=None)
    p.add_argument("--device",            default=None, help="cuda / cpu")
    # 新オプション
    p.add_argument("--use_amp",           action="store_true", default=None,
                   help="自動混合精度 (AMP) を有効化（CUDA 環境のみ）")
    p.add_argument("--patience",          type=int,   default=None,
                   help="Early stopping のエポック数（0=無効）")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None,
                   help="勾配蓄積ステップ数（デフォルト 1）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg: dict = json.load(f)

    # CLI 引数で設定を上書き
    for key in ["transformer_model", "num_epochs", "batch_size",
                "lr_transformer", "lr_task", "device",
                "patience", "gradient_accumulation_steps"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    # use_amp は store_true なので None チェック不要
    if args.use_amp:
        cfg["use_amp"] = True

    logger.info("Config: %s", json.dumps(cfg, indent=2, ensure_ascii=False))

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer_model"])

    # ---- Dataset ----
    train_ds = DyGIEDataset(
        path=args.train_path,
        tokenizer=tokenizer,
        max_span_width=cfg.get("max_span_width", 8),
        max_total_length=cfg.get("max_total_length", 512),
        use_ner=cfg.get("use_ner", True),
        use_rel=cfg.get("use_rel", True),
        use_coref=cfg.get("use_coref", True),
    )
    dev_ds = DyGIEDataset(
        path=args.dev_path,
        tokenizer=tokenizer,
        max_span_width=cfg.get("max_span_width", 8),
        max_total_length=cfg.get("max_total_length", 512),
        ner_labels=train_ds.ner_labels,
        rel_labels=train_ds.rel_labels,
        use_ner=cfg.get("use_ner", True),
        use_rel=cfg.get("use_rel", True),
        use_coref=cfg.get("use_coref", True),
    )

    batch_size = cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 0),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.get("num_workers", 0),
    )

    # ---- Model ----
    model = DyGIE(
        transformer_model=cfg["transformer_model"],
        ner_labels=train_ds.ner_labels,
        rel_labels=train_ds.rel_labels,
        max_span_width=cfg.get("max_span_width", 8),
        use_ner=cfg.get("use_ner", True),
        use_rel=cfg.get("use_rel", True),
        use_coref=cfg.get("use_coref", True),
        ner_loss_weight=cfg.get("ner_loss_weight", 1.0),
        rel_loss_weight=cfg.get("rel_loss_weight", 1.0),
        coref_loss_weight=cfg.get("coref_loss_weight", 1.0),
        width_embedding_dim=cfg.get("width_embedding_dim", 128),
        feedforward_dim=cfg.get("feedforward_dim", 150),
        use_attentive_pooling=cfg.get("use_attentive_pooling", True),
        spans_per_word=cfg.get("spans_per_word", 0.4),
        max_top_antecedents=cfg.get("max_top_antecedents", 50),
        dropout=cfg.get("dropout", 0.4),
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        output_dir=args.output_dir,
        num_epochs=cfg.get("num_epochs", 20),
        lr_transformer=cfg.get("lr_transformer", 1e-5),
        lr_task=cfg.get("lr_task", 1e-4),
        warmup_steps=cfg.get("warmup_steps", 200),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        use_gold_spans_for_rel=cfg.get("use_gold_spans_for_rel", True),
        device=cfg.get("device", None),
        log_every=cfg.get("log_every", 50),
        use_amp=cfg.get("use_amp", False),
        patience=cfg.get("patience", 0),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
    )

    # ラベル情報と設定を保存
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.json", "w") as f:
        json.dump(
            {"ner_labels": train_ds.ner_labels, "rel_labels": train_ds.rel_labels},
            f, indent=2,
        )
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    trainer.train()


if __name__ == "__main__":
    main()
