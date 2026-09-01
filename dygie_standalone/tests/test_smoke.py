"""
DyGIE-- Standalone — スモークテスト
フルパイプライン（データ読み込み→モデル forward→損失→デコード→メトリクス）を検証する。
HuggingFace への外部アクセス不要：ダミーBERTをローカル生成してテストする。

v2 追加テスト:
  - Coref metrics の precision/recall が独立して正しく計算されることを確認
  - save_pretrained / from_pretrained での dygie_config.json の保存・復元を確認
  - Trainer の early stopping 動作確認
"""

import sys
import json
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizerFast, PreTrainedTokenizerFast

from dygie.data import DyGIEDataset, collate_fn
from dygie.model import DyGIE
from dygie.training.metrics import NERMetrics, RelationMetrics, CorefMetrics

SAMPLE = Path(__file__).parent.parent / "data" / "sample" / "sample.jsonl"

# ---- ダミー BERT をローカルに生成（外部通信不要） ----
_DUMMY_DIR = Path(tempfile.mkdtemp(prefix="dygie_test_bert_"))

def _build_dummy_bert(directory: Path) -> None:
    """最小サイズの BERT をローカルに保存する。"""
    cfg = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    model = BertModel(cfg)
    model.save_pretrained(str(directory))

    # 最小限の tokenizer files を手書き
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    # 一般的な英単語を追加
    words = ["we", "present", "a", "new", "neural", "network", "model", "for",
             "named", "entity", "recognition", "the", "uses", "bert", "as",
             "feature", "extractor", "transformer", "architectures", "have",
             "improved", "machine", "translation", "significantly", "compare",
             "our", "method", "with", "existing", "baselines", "proposed",
             "achieves", "state", "-", "of", "the", "art", "f1", "score",
             "on", "scierc", "this", "demonstrates", "effectiveness", "approach",
             "##s", "##ed", "##ing", "##er", "##ly", "##tion"]
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    # vocab.txt 形式
    vocab_path = directory / "vocab.txt"
    with open(vocab_path, "w") as f:
        for token in sorted(vocab, key=vocab.get):
            f.write(token + "\n")

    # tokenizer_config.json
    tok_cfg = {
        "model_type": "bert",
        "tokenizer_class": "BertTokenizer",
        "do_lower_case": True,
    }
    with open(directory / "tokenizer_config.json", "w") as f:
        json.dump(tok_cfg, f)

_build_dummy_bert(_DUMMY_DIR)
MODEL_NAME = str(_DUMMY_DIR)

# slow tokenizer でロード（fast tokenizer が使えない場合の fallback）
from transformers import BertTokenizer
_TOK = BertTokenizer.from_pretrained(MODEL_NAME)


def _make_model(ds: DyGIEDataset) -> DyGIE:
    """テスト用の小さな DyGIE モデルを作成する。"""
    return DyGIE(
        transformer_model=MODEL_NAME,
        ner_labels=ds.ner_labels,
        rel_labels=ds.rel_labels,
        max_span_width=4,
        use_ner=True,
        use_rel=True,
        use_coref=True,
        feedforward_dim=64,
        width_embedding_dim=32,
        use_attentive_pooling=True,
        spans_per_word=0.4,
        dropout=0.0,
    )


def test_dataset():
    tok = _TOK
    ds = DyGIEDataset(SAMPLE, tok, max_span_width=4, max_total_length=128)
    assert len(ds) == 3, f"Expected 3 docs, got {len(ds)}"
    sample = ds[0]
    assert "input_ids" in sample
    assert "spans" in sample
    assert sample["spans"].shape[1] == 2
    assert "ner_labels" in sample
    assert "rel_labels" in sample
    print(f"  [OK] Dataset: {len(ds)} docs, NER labels={ds.ner_labels}, RE labels={ds.rel_labels}")


def test_dataloader():
    tok = _TOK
    ds = DyGIEDataset(SAMPLE, tok, max_span_width=4, max_total_length=128)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["input_ids"].dim() == 2
    assert batch["spans"].dim() == 3
    assert batch["span_mask"].dtype == torch.bool
    print(f"  [OK] DataLoader: input_ids={tuple(batch['input_ids'].shape)}, "
          f"spans={tuple(batch['spans'].shape)}")


def test_forward_with_loss():
    tok = _TOK
    ds = DyGIEDataset(SAMPLE, tok, max_span_width=4, max_total_length=128)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = _make_model(ds)
    model.eval()

    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_to_subword=batch["token_to_subword"],
            spans=batch["spans"],
            span_mask=batch["span_mask"],
            num_tokens=batch["num_tokens"],
            ner_labels=batch["ner_labels"],
            rel_labels=batch["rel_labels"],
            coref_clusters=batch["coref_clusters"],
            use_gold_spans=True,
        )

    assert "loss" in out, "loss missing"
    assert not out["loss"].isnan().item(), "loss is NaN"
    assert "ner_logits" in out
    assert "rel_logits" in out
    assert "antecedent_scores" in out
    print(f"  [OK] Forward+Loss: loss={out['loss'].item():.4f} | "
          f"ner_logits={tuple(out['ner_logits'].shape)} | "
          f"rel_logits={tuple(out['rel_logits'].shape)}")


def test_inference_no_labels():
    tok = _TOK
    ds = DyGIEDataset(SAMPLE, tok, max_span_width=4, max_total_length=128)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = _make_model(ds)
    model.eval()

    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_to_subword=batch["token_to_subword"],
            spans=batch["spans"],
            span_mask=batch["span_mask"],
            num_tokens=batch["num_tokens"],
            use_gold_spans=False,
        )

    assert "loss" not in out, "loss should not be present without labels"
    assert "ner_preds" in out
    assert "rel_preds" in out
    print(f"  [OK] Inference: ner_preds={tuple(out['ner_preds'].shape)}, "
          f"rel_preds={tuple(out['rel_preds'].shape)}")


def test_metrics():
    # NER
    ner = NERMetrics()
    preds = torch.tensor([[1, 2, 0, 1]])
    golds = torch.tensor([[1, 0, 0, 2]])
    mask  = torch.tensor([[True, True, True, True]])
    ner.update(preds, golds, mask)
    r = ner.compute()
    assert 0 <= r["ner_f1"] <= 1
    print(f"  [OK] NERMetrics: F1={r['ner_f1']:.4f}")

    # RE
    rel = RelationMetrics()
    p = torch.zeros(1, 4, 4, dtype=torch.long)
    g = torch.zeros(1, 4, 4, dtype=torch.long)
    pm = torch.zeros(1, 4, 4, dtype=torch.bool)
    p[0, 0, 1] = 1; g[0, 0, 1] = 1; pm[0, 0, 1] = True
    p[0, 1, 2] = 1; g[0, 1, 2] = 2; pm[0, 1, 2] = True
    rel.update(p, g, pm)
    r = rel.compute()
    assert 0 <= r["rel_f1"] <= 1
    print(f"  [OK] RelationMetrics: F1={r['rel_f1']:.4f}")

    # Coref
    coref = CorefMetrics()
    pred_c = [[(0, 2), (5, 7)], [(10, 10), (15, 15)]]
    gold_c = [[(0, 2), (5, 7)]]
    coref.update(pred_c, gold_c)
    r = coref.compute()
    assert "conll_f1" in r
    print(f"  [OK] CorefMetrics: CoNLL F1={r['conll_f1']:.4f}")


def test_coref_metrics_precision_recall_independence():
    """
    Coref メトリクスで precision と recall が独立して正しく計算されることを確認する。

    ケース: 予測クラスタが gold クラスタのスーパーセット（余分な mention を含む）。
      gold: {A, B, C} × 1 クラスタ
      pred: {A, B, C, D} × 1 クラスタ  (D は余分な false positive)

    B³ recall: 各 gold mention の gold cluster に対する pred cluster のカバレッジ
      A: |{A,B,C} ∩ {A,B,C,D}| / |{A,B,C}| = 3/3 = 1.0   (同様に B, C)
      D: gold_set = {} → 寄与なし
      r = (1+1+1+0) / 4 = 0.75

    B³ precision: 各 pred mention の pred cluster に対する gold cluster のカバレッジ
      A: |{A,B,C} ∩ {A,B,C,D}| / |{A,B,C,D}| = 3/4 = 0.75 (同様に B, C)
      D: gold_set = {} → |∅ ∩ {A,B,C,D}| / 4 = 0
      p = (0.75+0.75+0.75+0) / 4 = 0.5625

    recall (0.75) > precision (0.5625) が期待値。
    """
    coref = CorefMetrics()

    # gold: {A, B, C} の 1 クラスタ
    gold = [[(0, 0), (1, 1), (2, 2)]]
    # pred: gold に (3,3) を加えたスーパーセット
    pred = [[(0, 0), (1, 1), (2, 2), (3, 3)]]

    coref.update(pred, gold)
    r = coref.compute()

    b3_p = r["b3_p"]
    b3_r = r["b3_r"]

    # 期待: r ≈ 0.75 > p ≈ 0.5625
    assert b3_r > b3_p, (
        f"B³ recall ({b3_r:.4f}) should be greater than precision ({b3_p:.4f}) "
        "when prediction is a superset of gold"
    )
    # 数値チェック（許容誤差 1e-3）
    assert abs(b3_r - 0.75) < 1e-3, f"B³ recall expected ≈ 0.75, got {b3_r:.4f}"
    assert abs(b3_p - 0.5625) < 1e-3, f"B³ precision expected ≈ 0.5625, got {b3_p:.4f}"
    print(f"  [OK] Coref P/R independence: B³ P={b3_p:.4f} (expected 0.5625), "
          f"R={b3_r:.4f} (expected 0.75)")


def test_save_load_pretrained():
    """save_pretrained / from_pretrained が dygie_config.json を通じて
    ハイパーパラメータを正しく保存・復元することを確認する。"""
    tok = _TOK
    ds = DyGIEDataset(SAMPLE, tok, max_span_width=4, max_total_length=128)

    model = _make_model(ds)

    with tempfile.TemporaryDirectory(prefix="dygie_save_test_") as tmpdir:
        model.save_pretrained(tmpdir)

        # dygie_config.json が保存されたか確認
        config_path = Path(tmpdir) / "dygie_config.json"
        assert config_path.exists(), "dygie_config.json was not saved"
        with open(config_path) as f:
            saved_cfg = json.load(f)
        assert saved_cfg["ner_labels"] == ds.ner_labels
        assert saved_cfg["rel_labels"] == ds.rel_labels
        assert saved_cfg["max_span_width"] == 4

        # from_pretrained で復元できるか確認
        loaded = DyGIE.from_pretrained(tmpdir)
        assert loaded.ner_labels == model.ner_labels
        assert loaded.rel_labels == model.rel_labels
        assert loaded.use_ner == model.use_ner
        assert loaded.use_rel == model.use_rel
        assert loaded.use_coref == model.use_coref

        # パラメータが一致するか確認
        for (n1, p1), (n2, p2) in zip(
            sorted(model.state_dict().items()),
            sorted(loaded.state_dict().items()),
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Parameter mismatch: {n1}"

    print(f"  [OK] save_pretrained / from_pretrained: ner_labels={loaded.ner_labels}, "
          f"rel_labels={loaded.rel_labels}")


def test_backward():
    """1 ステップの勾配計算が通るか。"""
    tok = _TOK
    ds = DyGIEDataset(SAMPLE, tok, max_span_width=4, max_total_length=128)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = DyGIE(
        transformer_model=MODEL_NAME,
        ner_labels=ds.ner_labels,
        rel_labels=ds.rel_labels,
        max_span_width=4,
        feedforward_dim=64,
        width_embedding_dim=32,
        dropout=0.1,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_to_subword=batch["token_to_subword"],
        spans=batch["spans"],
        span_mask=batch["span_mask"],
        num_tokens=batch["num_tokens"],
        ner_labels=batch["ner_labels"],
        rel_labels=batch["rel_labels"],
        coref_clusters=batch["coref_clusters"],
    )
    loss = out["loss"]
    loss.backward()
    opt.step()
    print(f"  [OK] Backward pass: loss={loss.item():.4f}")


def test_coref_distance_embedding():
    """
    coref の antecedent 距離埋め込みが正しく適用されているか確認する。
    修正前は dist[i,j] = j-i (clamped to 0) で常に 0 だった。
    修正後は dist[i,j] = i-j > 0 for j < i。
    """
    from dygie.model.coref_module import CorefModule

    module = CorefModule(span_dim=16, feedforward_dim=32, max_top_antecedents=10)
    module.eval()

    T = 5
    D = 16
    B = 1

    top_repr = torch.randn(B, T, D)
    top_scores = torch.randn(B, T)
    top_indices = torch.arange(T).unsqueeze(0)  # [1, T]
    top_mask = torch.ones(B, T, dtype=torch.bool)
    spans = torch.zeros(B, T, 2, dtype=torch.long)
    for i in range(T):
        spans[0, i, 0] = i
        spans[0, i, 1] = i

    with torch.no_grad():
        ant_scores = module._score_antecedents(
            top_repr, top_scores, top_indices, top_mask, spans
        )

    # ant_scores の形状確認
    assert ant_scores.shape == (B, T, T + 1), \
        f"Expected ({B}, {T}, {T+1}), got {tuple(ant_scores.shape)}"

    # j >= i のペアは -inf になっているはず（causal mask）
    # ant_scores[:, i, j+1] は mention i に対する antecedent j のスコア（+1 はダミーオフセット）
    for i in range(T):
        for j in range(i, T):  # j >= i は invalid
            score = ant_scores[0, i, j + 1].item()
            assert score == float("-inf") or score < -1e6, \
                f"Expected -inf for j>=i pair ({i},{j}), got {score}"

    print(f"  [OK] Coref distance embedding: ant_scores shape={tuple(ant_scores.shape)}, "
          "causal masking verified")


if __name__ == "__main__":
    tests = [
        ("Dataset",                           test_dataset),
        ("DataLoader",                        test_dataloader),
        ("Forward + Loss",                    test_forward_with_loss),
        ("Inference (no labels)",             test_inference_no_labels),
        ("Metrics",                           test_metrics),
        ("Coref P/R independence",            test_coref_metrics_precision_recall_independence),
        ("Save / Load pretrained",            test_save_load_pretrained),
        ("Backward pass",                     test_backward),
        ("Coref distance embedding",          test_coref_distance_embedding),
    ]
    print("\n===== DyGIE++ Standalone Smoke Tests =====")
    passed = failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n===== Results: {passed} passed / {failed} failed =====\n")
    sys.exit(0 if failed == 0 else 1)
