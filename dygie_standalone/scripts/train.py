#!/usr/bin/env python3
"""
DyGIE++ Standalone — 学習スクリプト

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
    p.add_argument("--early_stopping_warmup", type=int, default=None,
                   help="Early stopping を開始するまでのウォームアップエポック数（デフォルト 20）")
    p.add_argument("--gradient_accumulation_steps", type=int, default=None,
                   help="勾配蓄積ステップ数（デフォルト 1）")
    # メモリ最適化オプション
    p.add_argument("--use_gradient_checkpointing", action="store_true", default=None,
                   help="Transformer エンコーダに勾配チェックポイントを適用（メモリ削減）")
    p.add_argument("--max_spans",         type=int,   default=None,
                   help="文書あたりの最大スパン数（0=上限なし）。RE の K×K メモリを削減する。")
    # RE スコア改善オプション (v4)
    p.add_argument("--type_embedding_dim", type=int,   default=None,
                   help="RE エンティティタイプ埋め込みの次元数（0=無効）")
    p.add_argument("--use_distance_feature", action="store_true", default=None,
                   help="RE スパン間距離特徴を有効化")
    p.add_argument("--num_distance_buckets", type=int,   default=None,
                   help="距離バケット数（デフォルト 10）")
    p.add_argument("--distance_embedding_dim", type=int,   default=None,
                   help="距離埋め込みの次元数（0=無効）")
    p.add_argument("--focal_loss_gamma",  type=float, default=None,
                   help="RE Focal Loss の gamma 値（0=通常の CE、2.0 が推奨）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg: dict = json.load(f)

    # CLI 引数で設定を上書き
    for key in ["transformer_model", "num_epochs", "batch_size",
                "lr_transformer", "lr_task", "device",
                "patience", "early_stopping_warmup", "gradient_accumulation_steps", "max_spans",
                "type_embedding_dim", "num_distance_buckets",
                "distance_embedding_dim", "focal_loss_gamma"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    # store_true フラグは None チェック不要
    if args.use_amp:
        cfg["use_amp"] = True
    if args.use_gradient_checkpointing:
        cfg["use_gradient_checkpointing"] = True
    if args.use_distance_feature:
        cfg["use_distance_feature"] = True

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
        max_spans=cfg.get("max_spans", 0),
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
        max_spans=cfg.get("max_spans", 0),
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
        use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", False),
        # v4: RE スコア改善
        type_embedding_dim=cfg.get("type_embedding_dim", 0),
        use_distance_feature=cfg.get("use_distance_feature", False),
        num_distance_buckets=cfg.get("num_distance_buckets", 10),
        distance_embedding_dim=cfg.get("distance_embedding_dim", 64),
        focal_loss_gamma=cfg.get("focal_loss_gamma", 0.0),
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
        early_stopping_warmup=cfg.get("early_stopping_warmup", 20),
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
