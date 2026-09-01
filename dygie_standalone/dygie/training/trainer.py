"""
DyGIE-- Standalone — Trainer

AllenNLP の Trainer を置き換える純粋な PyTorch 学習ループ。

特徴:
  - Linear warmup + linear decay スケジューラ
  - Transformer と task head で異なる学習率（layerwise LR decay もオプション）
  - Gradient clipping
  - 最良モデルのチェックポイント保存
  - TensorBoard / tqdm によるログ（オプション）

改善点 (v2):
  - AMP (自動混合精度) 学習サポート (use_amp=True, CUDA 環境のみ)
  - Early stopping サポート (patience > 0 で有効)
  - Gradient accumulation サポート (gradient_accumulation_steps > 1 で有効)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..model.dygie import DyGIE
from .metrics import NERMetrics, RelationMetrics, CorefMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Parameters
    ----------
    model : DyGIE
    train_loader : DataLoader
    dev_loader : DataLoader
    output_dir : str | Path
    num_epochs : int
    lr_transformer : float
        Transformer エンコーダの学習率。
    lr_task : float
        タスクヘッドの学習率。
    warmup_steps : int
    max_grad_norm : float
    use_gold_spans_for_rel : bool
        学習中に gold NER スパンを RE に使用するか（True 推奨）。
    device : str | None
        "cuda", "cpu", または None（自動選択）。
    log_every : int
        何ステップごとに損失をログ出力するか。
    use_amp : bool
        自動混合精度 (AMP) を使用するか。CUDA 環境でのみ有効。
        学習速度と显存効率が向上する。
    patience : int
        Early stopping の待機エポック数。0 で無効（常に全エポック学習）。
        dev スコアが patience エポック改善しない場合に学習を停止する。
    gradient_accumulation_steps : int
        勾配蓄積ステップ数。1 で通常通り毎ステップ更新。
        メモリが少ない環境で実効バッチサイズを増やすために使用する。
    """

    def __init__(
        self,
        model: DyGIE,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        output_dir: str | Path,
        num_epochs: int = 20,
        lr_transformer: float = 1e-5,
        lr_task: float = 1e-4,
        warmup_steps: int = 200,
        max_grad_norm: float = 1.0,
        use_gold_spans_for_rel: bool = True,
        device: str | None = None,
        log_every: int = 50,
        use_amp: bool = False,
        patience: int = 0,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.use_gold_spans_for_rel = use_gold_spans_for_rel
        self.log_every = log_every
        self.patience = patience
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info("Training on device: %s", self.device)

        # AMP (自動混合精度): CUDA のみ有効
        self.use_amp = use_amp and self.device.type == "cuda"
        if use_amp and not self.use_amp:
            logger.warning("AMP requested but device is not CUDA. AMP disabled.")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("AMP (automatic mixed precision) enabled.")

        # optimizer: Transformer と task head を分離
        encoder_params = list(model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        task_params = [p for p in model.parameters() if id(p) not in encoder_ids]

        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": lr_transformer},
                {"params": task_params,    "lr": lr_task},
            ],
            weight_decay=1e-2,
        )

        # スケジューラのステップ数は gradient_accumulation_steps を考慮
        effective_steps_per_epoch = max(
            1, len(train_loader) // self.gradient_accumulation_steps
        )
        total_steps = effective_steps_per_epoch * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # metrics
        self.ner_metrics = NERMetrics()
        self.rel_metrics = RelationMetrics()
        self.coref_metrics = CorefMetrics()

        self.best_dev_score = -1.0
        self._patience_counter = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            dev_metrics = self._evaluate(self.dev_loader)
            elapsed = time.time() - t0

            # 主要スコア（NER F1 を優先、なければ RE F1）
            score = dev_metrics.get("ner_f1", dev_metrics.get("rel_f1", 0.0))

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "elapsed": elapsed,
                **dev_metrics,
            }
            self.history.append(record)
            self._log_epoch(record)

            # チェックポイント
            if score >= self.best_dev_score:
                self.best_dev_score = score
                self._patience_counter = 0
                self._save_checkpoint("best")
                logger.info("  ↑ New best score: %.4f", score)
            else:
                self._patience_counter += 1
                logger.info(
                    "  No improvement. Patience: %d/%d",
                    self._patience_counter,
                    self.patience if self.patience > 0 else float("inf"),
                )

            self._save_checkpoint("last")

            # Early stopping
            if self.patience > 0 and self._patience_counter >= self.patience:
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d). "
                    "Best score: %.4f",
                    epoch, self.patience, self.best_dev_score,
                )
                break

        # 学習履歴の保存
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training complete. Best dev score: %.4f", self.best_dev_score)

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return self._evaluate(loader)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_to_subword=batch["token_to_subword"],
                    spans=batch["spans"],
                    span_mask=batch["span_mask"],
                    num_tokens=batch["num_tokens"],
                    ner_labels=batch.get("ner_labels"),
                    rel_labels=batch.get("rel_labels"),
                    coref_clusters=batch.get("coref_clusters"),
                    use_gold_spans=self.use_gold_spans_for_rel,
                )

            loss: torch.Tensor = outputs["loss"]

            # gradient accumulation: 損失をステップ数で割って正規化
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            # gradient_accumulation_steps ごとにパラメータを更新
            is_update_step = (step + 1) % self.gradient_accumulation_steps == 0
            is_last_step = (step + 1) == len(self.train_loader)

            if is_update_step or is_last_step:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # 元の損失（正規化前）を記録
            raw_loss = outputs["loss"].item()
            total_loss += raw_loss
            n_steps += 1

            if step % self.log_every == 0:
                ner_l = outputs.get("ner_loss", torch.tensor(0.0)).item()
                rel_l = outputs.get("rel_loss", torch.tensor(0.0)).item()
                coref_l = outputs.get("coref_loss", torch.tensor(0.0)).item()
                logger.info(
                    "Epoch %d step %d/%d | loss=%.4f ner=%.4f rel=%.4f coref=%.4f%s",
                    epoch, step, len(self.train_loader),
                    raw_loss, ner_l, rel_l, coref_l,
                    " [AMP]" if self.use_amp else "",
                )

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        self.ner_metrics.reset()
        self.rel_metrics.reset()
        self.coref_metrics.reset()

        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_to_subword=batch["token_to_subword"],
                spans=batch["spans"],
                span_mask=batch["span_mask"],
                num_tokens=batch["num_tokens"],
                use_gold_spans=False,  # 推論時は predicted NER を使用
            )

            if self.model.use_ner and "ner_preds" in outputs:
                self.ner_metrics.update(
                    preds=outputs["ner_preds"].cpu(),
                    golds=batch["ner_labels"].cpu(),
                    span_mask=batch["span_mask"].cpu(),
                )

            if self.model.use_rel and "rel_preds" in outputs:
                self.rel_metrics.update(
                    preds=outputs["rel_preds"].cpu(),
                    golds=batch["rel_labels"].cpu(),
                    pair_mask=outputs["pair_mask"].cpu(),
                )

            if self.model.use_coref and "top_span_indices" in outputs:
                for b in range(batch["input_ids"].size(0)):
                    pred_clusters = self._decode_coref(
                        outputs["antecedent_scores"][b],
                        outputs["top_span_indices"][b],
                        outputs["top_span_mask"][b],
                        batch["spans"][b],
                    )
                    self.coref_metrics.update(
                        pred_clusters=pred_clusters,
                        gold_clusters=batch["coref_clusters"][b],
                    )

        metrics: dict[str, float] = {}
        if self.model.use_ner:
            metrics.update(self.ner_metrics.compute())
        if self.model.use_rel:
            metrics.update(self.rel_metrics.compute())
        if self.model.use_coref:
            metrics.update(self.coref_metrics.compute())

        return metrics

    def _decode_coref(
        self,
        ant_scores: torch.Tensor,    # [T, T+1]
        top_indices: torch.Tensor,   # [T]
        top_mask: torch.Tensor,      # [T]
        spans: torch.Tensor,         # [K, 2]
    ) -> list[list[tuple[int, int]]]:
        """antecedent スコアから予測クラスタを構築する。"""
        T = ant_scores.size(0)
        # 各 mention の best antecedent
        best_ants = ant_scores.argmax(dim=-1)  # [T] 0=ダミー

        # Union-Find でクラスタ化
        parent = list(range(T))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i in range(T):
            if not top_mask[i].item():
                continue
            ant = best_ants[i].item() - 1  # -1 でダミーオフセット除去
            if ant >= 0 and top_mask[ant].item():
                union(i, ant)

        # クラスタ収集
        cluster_map: dict[int, list[tuple[int, int]]] = {}
        for i in range(T):
            if not top_mask[i].item():
                continue
            root = find(i)
            span_idx = top_indices[i].item()
            span = (spans[span_idx, 0].item(), spans[span_idx, 1].item())
            cluster_map.setdefault(root, []).append(span)

        return [c for c in cluster_map.values() if len(c) > 1]

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = self.output_dir / f"checkpoint_{tag}"
        self.model.save_pretrained(ckpt_dir)
        # ラベル情報も保存（後方互換のため）
        label_info = {
            "ner_labels": self.model.ner_labels,
            "rel_labels": self.model.rel_labels,
        }
        with open(ckpt_dir / "labels.json", "w") as f:
            json.dump(label_info, f, indent=2)

    @staticmethod
    def _log_epoch(record: dict[str, Any]) -> None:
        parts = [f"Epoch {record['epoch']}"]
        parts.append(f"train_loss={record['train_loss']:.4f}")
        for k, v in record.items():
            if k not in ("epoch", "train_loss", "elapsed") and isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
        parts.append(f"({record['elapsed']:.1f}s)")
        logger.info(" | ".join(parts))
