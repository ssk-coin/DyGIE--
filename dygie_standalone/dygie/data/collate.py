"""
DyGIE-- Standalone — Collate
可変長テンソルをパディングしてバッチ化します。
スパン数・トークン数がドキュメントごとに異なるため、
各フィールドをゼロパディングして揃えます。
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    DataLoader の collate_fn。

    各サンプルは dataset.py の _process() が返す dict。
    可変長フィールドをパディングし、まとめた dict を返す。

    Returns
    -------
    dict with keys:
      doc_keys         : list[str]
      input_ids        : LongTensor [B, L_max]
      attention_mask   : LongTensor [B, L_max]
      token_to_subword : LongTensor [B, N_max]
      subword_to_token : LongTensor [B, L_max]
      sentence_offsets : list[LongTensor [S_i, 2]]  (パディング不可・リストのまま)
      spans            : LongTensor [B, K_max, 2]
      span_mask        : BoolTensor [B, K_max]       有効スパンのマスク
      ner_labels       : LongTensor [B, K_max]
      rel_labels       : LongTensor [B, K_max, K_max]
      coref_clusters   : list[list[list[tuple]]]     (リストのまま)
      num_tokens       : LongTensor [B]
    """
    doc_keys = [s["doc_key"] for s in batch]
    num_tokens = torch.tensor([s["num_tokens"] for s in batch], dtype=torch.long)

    # ---- sequence padding ----
    input_ids = pad_sequence(
        [s["input_ids"] for s in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [s["attention_mask"] for s in batch], batch_first=True, padding_value=0
    )
    subword_to_token = pad_sequence(
        [s["subword_to_token"] for s in batch], batch_first=True, padding_value=-1
    )
    token_to_subword = pad_sequence(
        [s["token_to_subword"] for s in batch], batch_first=True, padding_value=0
    )

    # ---- span padding ----
    max_spans = max(s["spans"].size(0) for s in batch)
    B = len(batch)

    spans = torch.zeros(B, max_spans, 2, dtype=torch.long)
    span_mask = torch.zeros(B, max_spans, dtype=torch.bool)
    ner_labels = torch.zeros(B, max_spans, dtype=torch.long)
    rel_labels = torch.zeros(B, max_spans, max_spans, dtype=torch.long)

    for i, s in enumerate(batch):
        K = s["spans"].size(0)
        spans[i, :K] = s["spans"]
        span_mask[i, :K] = True
        ner_labels[i, :K] = s["ner_labels"]
        rel_labels[i, :K, :K] = s["rel_labels"]

    sentence_offsets = [s["sentence_offsets"] for s in batch]
    coref_clusters = [s["coref_clusters"] for s in batch]

    return {
        "doc_keys": doc_keys,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_to_subword": token_to_subword,
        "subword_to_token": subword_to_token,
        "sentence_offsets": sentence_offsets,
        "spans": spans,
        "span_mask": span_mask,
        "ner_labels": ner_labels,
        "rel_labels": rel_labels,
        "coref_clusters": coref_clusters,
        "num_tokens": num_tokens,
    }
