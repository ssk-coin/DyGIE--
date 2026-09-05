#!/usr/bin/env python3
"""
DyGIE-- — 推論スクリプト

使い方:
  python scripts/predict.py \
      --model_dir output/scierc_bert/checkpoint_best \
      --input_path data/test.jsonl \
      --output_path predictions.jsonl
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from dygie.data import DyGIEDataset, collate_fn
from dygie.model import DyGIE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DyGIE++ Standalone Prediction")
    p.add_argument("--model_dir",   required=True)
    p.add_argument("--input_path",  required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--device",      default=None)
    return p.parse_args()


def _decode_coref(
    ant_scores: torch.Tensor,
    top_indices: torch.Tensor,
    top_mask: torch.Tensor,
    spans: torch.Tensor,
) -> list[list[tuple[int, int]]]:
    T = ant_scores.size(0)
    best_ants = ant_scores.argmax(dim=-1)
    parent = list(range(T))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(T):
        if not top_mask[i].item():
            continue
        ant = best_ants[i].item() - 1
        if ant >= 0 and top_mask[ant].item():
            parent[find(i)] = find(ant)

    cluster_map: dict[int, list] = {}
    for i in range(T):
        if not top_mask[i].item():
            continue
        root = find(i)
        idx = top_indices[i].item()
        span = [spans[idx, 0].item(), spans[idx, 1].item()]
        cluster_map.setdefault(root, []).append(span)

    return [c for c in cluster_map.values() if len(c) > 1]


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # ---- dygie_config.json があれば from_pretrained で一括ロード ----
    dygie_config_path = model_dir / "dygie_config.json"
    if dygie_config_path.exists():
        logger.info("Loading model config from dygie_config.json")
        with open(dygie_config_path, encoding="utf-8") as f:
            dygie_cfg: dict = json.load(f)

        ner_labels: list[str] = dygie_cfg["ner_labels"]
        rel_labels: list[str] = dygie_cfg["rel_labels"]

        # tokenizer は encoder の名前 / ローカルパスから
        # dygie_config には元のモデル名が入っているので親ディレクトリの config.json も参照
        tok_model = dygie_cfg.get("transformer_model", str(model_dir))
        # もし config.json に元のモデル名があればそちらを優先
        config_path = model_dir / "config.json"
        if not config_path.exists():
            config_path = model_dir.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved_cfg: dict = json.load(f)
            # saved_cfg["transformer_model"] は元の HuggingFace モデル名
            if "transformer_model" in saved_cfg:
                tok_model = saved_cfg["transformer_model"]

        tokenizer = AutoTokenizer.from_pretrained(tok_model)
        model = DyGIE.from_pretrained(model_dir)
    else:
        # 旧形式: config.json + labels.json で手動ロード
        logger.info("dygie_config.json not found, loading config from config.json + labels.json")

        config_path = model_dir / "config.json"
        if not config_path.exists():
            config_path = model_dir.parent / "config.json"
        with open(config_path) as f:
            cfg: dict = json.load(f)

        labels_path = model_dir / "labels.json"
        if not labels_path.exists():
            labels_path = model_dir.parent / "labels.json"
        with open(labels_path) as f:
            labels: dict = json.load(f)

        ner_labels = labels["ner_labels"]
        rel_labels = labels["rel_labels"]
        dygie_cfg = cfg  # fallback

        tokenizer = AutoTokenizer.from_pretrained(cfg["transformer_model"])
        model = DyGIE(
            transformer_model=cfg["transformer_model"],
            ner_labels=ner_labels,
            rel_labels=rel_labels,
            max_span_width=cfg.get("max_span_width", 8),
            use_ner=cfg.get("use_ner", True),
            use_rel=cfg.get("use_rel", True),
            use_coref=cfg.get("use_coref", True),
            width_embedding_dim=cfg.get("width_embedding_dim", 128),
            feedforward_dim=cfg.get("feedforward_dim", 150),
            use_attentive_pooling=cfg.get("use_attentive_pooling", True),
            spans_per_word=cfg.get("spans_per_word", 0.4),
            max_top_antecedents=cfg.get("max_top_antecedents", 50),
            dropout=cfg.get("dropout", 0.4),
        )
        state = torch.load(
            model_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    ner_id2label = {i + 1: lbl for i, lbl in enumerate(ner_labels)}
    rel_id2label = {i + 1: lbl for i, lbl in enumerate(rel_labels)}

    # ---- Dataset ----
    dataset = DyGIEDataset(
        path=args.input_path,
        tokenizer=tokenizer,
        max_span_width=dygie_cfg.get("max_span_width", 8),
        max_total_length=dygie_cfg.get("max_total_length", 512),
        ner_labels=ner_labels,
        rel_labels=rel_labels,
        use_ner=dygie_cfg.get("use_ner", True),
        use_rel=dygie_cfg.get("use_rel", True),
        use_coref=dygie_cfg.get("use_coref", True),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    predictions: list[dict] = []
    doc_idx = 0

    with torch.no_grad():
        for batch in loader:
            bsz = batch["input_ids"].size(0)
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
                token_to_subword=batch_device["token_to_subword"],
                spans=batch_device["spans"],
                span_mask=batch_device["span_mask"],
                num_tokens=batch_device["num_tokens"],
                use_gold_spans=False,
            )

            for b in range(bsz):
                raw_doc = dataset.raw_docs[doc_idx]
                mask = batch["span_mask"][b]  # CPU
                spans_b = batch["spans"][b]   # CPU

                pred_doc: dict = {
                    "doc_key": batch["doc_keys"][b],
                    "dataset": raw_doc.get("dataset", ""),
                    "sentences": raw_doc["sentences"],
                }

                # NER 予測
                if model.use_ner and "ner_preds" in outputs:
                    ner_preds = outputs["ner_preds"][b].cpu()
                    pred_ner_flat: list[list] = []
                    for i, (valid, label_id) in enumerate(
                        zip(mask.tolist(), ner_preds.tolist())
                    ):
                        if valid and label_id > 0:
                            s, e = spans_b[i, 0].item(), spans_b[i, 1].item()
                            pred_ner_flat.append([s, e, ner_id2label[label_id]])
                    # 文ごとに分割
                    sent_offsets = batch["sentence_offsets"][b]
                    pred_ner_by_sent: list[list] = [[] for _ in raw_doc["sentences"]]
                    for annot in pred_ner_flat:
                        for si, (ss, se) in enumerate(sent_offsets.tolist()):
                            if ss <= annot[0] <= se:
                                pred_ner_by_sent[si].append(annot)
                                break
                    pred_doc["predicted_ner"] = pred_ner_by_sent

                # RE 予測
                if model.use_rel and "rel_preds" in outputs:
                    rel_preds = outputs["rel_preds"][b].cpu()
                    pair_mask = outputs.get("pair_mask")
                    pm_b = pair_mask[b].cpu() if pair_mask is not None else None
                    pred_rel_flat: list[list] = []
                    K = spans_b.size(0)
                    for i in range(K):
                        for j in range(K):
                            if pm_b is not None and not pm_b[i, j].item():
                                continue
                            label_id = rel_preds[i, j].item()
                            if label_id > 0:
                                s1, e1 = spans_b[i, 0].item(), spans_b[i, 1].item()
                                s2, e2 = spans_b[j, 0].item(), spans_b[j, 1].item()
                                pred_rel_flat.append(
                                    [s1, e1, s2, e2, rel_id2label[label_id]]
                                )
                    sent_offsets = batch["sentence_offsets"][b]
                    pred_rel_by_sent: list[list] = [[] for _ in raw_doc["sentences"]]
                    for annot in pred_rel_flat:
                        for si, (ss, se) in enumerate(sent_offsets.tolist()):
                            if ss <= annot[0] <= se:
                                pred_rel_by_sent[si].append(annot)
                                break
                    pred_doc["predicted_relations"] = pred_rel_by_sent

                # Coref 予測
                if model.use_coref and "top_span_indices" in outputs:
                    pred_clusters = _decode_coref(
                        outputs["antecedent_scores"][b].cpu(),
                        outputs["top_span_indices"][b].cpu(),
                        outputs["top_span_mask"][b].cpu(),
                        spans_b,
                    )
                    pred_doc["predicted_clusters"] = pred_clusters

                predictions.append(pred_doc)
                doc_idx += 1

    with open(args.output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    logger.info("Wrote %d predictions to %s", len(predictions), args.output_path)


if __name__ == "__main__":
    main()
