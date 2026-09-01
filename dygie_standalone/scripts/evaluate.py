#!/usr/bin/env python3
"""
DyGIE++ Standalone — 評価スクリプト

Gold と Prediction の jsonl を比較して NER / RE / Coref の F1 を計算します。

使い方:
  python scripts/evaluate.py \
      --gold_path data/test.jsonl \
      --pred_path predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dygie.training.metrics import NERMetrics, RelationMetrics, CorefMetrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gold_path", required=True)
    p.add_argument("--pred_path", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gold_docs: dict[str, dict] = {}
    with open(args.gold_path) as f:
        for line in f:
            doc = json.loads(line)
            gold_docs[doc["doc_key"]] = doc

    pred_docs: dict[str, dict] = {}
    with open(args.pred_path) as f:
        for line in f:
            doc = json.loads(line)
            pred_docs[doc["doc_key"]] = doc

    ner_tp = ner_fp = ner_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    coref_metrics = CorefMetrics()

    for doc_key, pred in pred_docs.items():
        gold = gold_docs.get(doc_key, {})

        # ---- NER ----
        gold_ner: set[tuple] = set()
        for sent_ner in gold.get("ner", []):
            for s, e, lbl in sent_ner:
                gold_ner.add((s, e, lbl))

        pred_ner: set[tuple] = set()
        for sent_ner in pred.get("predicted_ner", []):
            for s, e, lbl in sent_ner:
                pred_ner.add((s, e, lbl))

        ner_tp += len(gold_ner & pred_ner)
        ner_fp += len(pred_ner - gold_ner)
        ner_fn += len(gold_ner - pred_ner)

        # ---- RE ----
        gold_rel: set[tuple] = set()
        for sent_rel in gold.get("relations", []):
            for s1, e1, s2, e2, lbl in sent_rel:
                gold_rel.add((s1, e1, s2, e2, lbl))

        pred_rel: set[tuple] = set()
        for sent_rel in pred.get("predicted_relations", []):
            for s1, e1, s2, e2, lbl in sent_rel:
                pred_rel.add((s1, e1, s2, e2, lbl))

        rel_tp += len(gold_rel & pred_rel)
        rel_fp += len(pred_rel - gold_rel)
        rel_fn += len(gold_rel - pred_rel)

        # ---- Coref ----
        gold_clusters = [
            [tuple(m) for m in cluster]
            for cluster in gold.get("clusters", [])
        ]
        pred_clusters = [
            [tuple(m) for m in cluster]
            for cluster in pred.get("predicted_clusters", [])
        ]
        if gold_clusters or pred_clusters:
            coref_metrics.update(
                pred_clusters=pred_clusters,
                gold_clusters=gold_clusters,
            )

    # ---- 出力 ----
    def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f = 2 * p * r / (p + r + 1e-9)
        return p, r, f

    ner_p, ner_r, ner_f = _f1(ner_tp, ner_fp, ner_fn)
    rel_p, rel_r, rel_f = _f1(rel_tp, rel_fp, rel_fn)

    print("\n===== Evaluation Results =====")
    print(f"NER  | P={ner_p:.4f}  R={ner_r:.4f}  F1={ner_f:.4f}"
          f"  (TP={ner_tp} FP={ner_fp} FN={ner_fn})")
    print(f"RE   | P={rel_p:.4f}  R={rel_r:.4f}  F1={rel_f:.4f}"
          f"  (TP={rel_tp} FP={rel_fp} FN={rel_fn})")

    coref_result = coref_metrics.compute()
    if coref_result.get("conll_f1", 0.0) > 0:
        print(f"Coref MUC   | P={coref_result['muc_p']:.4f}  "
              f"R={coref_result['muc_r']:.4f}  F1={coref_result['muc_f1']:.4f}")
        print(f"Coref B³    | P={coref_result['b3_p']:.4f}  "
              f"R={coref_result['b3_r']:.4f}  F1={coref_result['b3_f1']:.4f}")
        print(f"Coref CEAFφ4| P={coref_result['ceaf_p']:.4f}  "
              f"R={coref_result['ceaf_r']:.4f}  F1={coref_result['ceaf_f1']:.4f}")
        print(f"CoNLL Score | F1={coref_result['conll_f1']:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()
