#!/usr/bin/env python3
"""
DyGIE-- Standalone — 評価スクリプト

Gold と Prediction の jsonl を比較して NER / RE / Coref の F1 を計算します。

使い方:
  # 基本（フルデータ評価）
  python scripts/evaluate.py \
      --gold_path data/test.jsonl \
      --pred_path predictions.jsonl

  # モデルの予測空間内に制限した評価（trainer の表示値と一致する）
  python scripts/evaluate.py \
      --gold_path data/test.jsonl \
      --pred_path predictions.jsonl \
      --max_span_width 8

  # モデルディレクトリからパラメータを自動読み込み
  python scripts/evaluate.py \
      --gold_path data/test.jsonl \
      --pred_path predictions.jsonl \
      --model_dir output/scierc_bert/checkpoint_best

備考:
  --max_span_width を指定すると、gold アノテーションのうちスパン幅が
  max_span_width を超えるものを評価から除外します（モデルが予測不可能なスパン）。
  これにより trainer が学習中に表示するスコアと evaluate.py の結果が一致します。
  指定しない場合（デフォルト 0）はフィルタリングなし（従来の厳格評価）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dygie.training.metrics import NERMetrics, RelationMetrics, CorefMetrics
from dygie.version import print_version_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DyGIE++ Standalone Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gold_path", required=True, help="Gold データ (.jsonl)")
    p.add_argument("--pred_path", required=True, help="予測データ (.jsonl)")
    p.add_argument(
        "--max_span_width",
        type=int,
        default=0,
        help="最大スパン幅（トークン数）。指定すると gold アノテーションをこの幅以下に制限する。"
             "0=無制限（デフォルト）。trainer と同じ値（通常 8）を指定すると"
             "trainer の表示スコアと一致した評価になる。",
    )
    p.add_argument(
        "--model_dir",
        default=None,
        help="モデルディレクトリ。dygie_config.json / config.json から"
             "max_span_width を自動読み込みする（--max_span_width より優先）。",
    )
    return p.parse_args()


def _load_max_span_width(model_dir: str) -> int | None:
    """モデルディレクトリから max_span_width を読み込む。"""
    d = Path(model_dir)
    for fname in ("dygie_config.json", "config.json"):
        cfg_path = d / fname
        if not cfg_path.exists():
            cfg_path = d.parent / fname
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            if "max_span_width" in cfg:
                return int(cfg["max_span_width"])
    return None


def _span_in_range(s: int, e: int, max_span_width: int) -> bool:
    """スパン (s, e) がモデルの予測空間内かどうかを返す。

    DyGIE++ のスパン列挙では end は inclusive で、
    span width = e - s + 1 ≤ max_span_width となる。
    """
    return (e - s + 1) <= max_span_width


def main() -> None:
    # バージョン情報をログ冒頭に表示（実験結果と対応付けるため）
    print_version_header()

    args = parse_args()

    # max_span_width の決定
    max_span_width = 0
    if args.model_dir is not None:
        loaded_msw = _load_max_span_width(args.model_dir)
        if loaded_msw is not None:
            max_span_width = loaded_msw
            print(f"[INFO] model_dir から max_span_width={max_span_width} を読み込みました。")
        else:
            print(f"[WARN] model_dir から max_span_width を読み込めませんでした。"
                  f"フィルタリングなしで評価します。")
    if args.max_span_width > 0:
        max_span_width = args.max_span_width  # CLI 引数が model_dir より優先

    if max_span_width > 0:
        print(f"[INFO] gold アノテーションをスパン幅 ≤ {max_span_width} トークンに制限します。")
    else:
        print("[INFO] スパン幅フィルタリングなし（全 gold アノテーションを評価対象とします）。")

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
                # max_span_width フィルタ
                if max_span_width > 0 and not _span_in_range(s, e, max_span_width):
                    continue
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
                # 両エンティティスパンが max_span_width 内かチェック
                if max_span_width > 0 and (
                    not _span_in_range(s1, e1, max_span_width)
                    or not _span_in_range(s2, e2, max_span_width)
                ):
                    continue
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

    msw_note = f" [max_span_width={max_span_width}]" if max_span_width > 0 else ""
    print(f"\n===== Evaluation Results{msw_note} =====")
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
