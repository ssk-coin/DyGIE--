# DyGIE++ Standalone

AllenNLP **不要**の DyGIE++ 完全再実装です。  
PyTorch + HuggingFace Transformers のみで動作します。

---

## 目次

1. [概要](#概要)
2. [対応タスク](#対応タスク)
3. [必要環境・インストール](#必要環境インストール)
4. [ディレクトリ構成](#ディレクトリ構成)
5. [データ形式](#データ形式)
6. [クイックスタート](#クイックスタート)
7. [学習](#学習)
8. [推論](#推論)
9. [評価](#評価)
10. [モデルの保存と復元](#モデルの保存と復元)
11. [Python API](#python-api)
12. [テスト](#テスト)
13. [ハイパーパラメータ一覧](#ハイパーパラメータ一覧)
14. [バグ修正と改善点 (v2)](#バグ修正と改善点-v2)
15. [トラブルシューティング](#トラブルシューティング)

---

## 概要

DyGIE++ (Wadden et al., 2019) は、Transformer エンコーダをバックボーンとする情報抽出モデルです。  
固有表現認識 (NER)・関係抽出 (RE)・共参照解析 (Coref) を統一的なスパンベースの枠組みで扱います。

本実装の特徴:

- **AllenNLP ゼロ依存** — PyTorch + Transformers だけで動作
- **SciERC / DyGIE++ JSON Lines 形式**をそのまま入力可能
- **AMP（自動混合精度）・Early stopping・Gradient accumulation** をサポート（v2）
- **`save_pretrained` / `from_pretrained`** による簡単なモデル保存・復元（v2）

---

## 対応タスク

| タスク | 概要 |
|--------|------|
| **NER** | Named Entity Recognition — スパン分類 |
| **RE**  | Relation Extraction — スパンペア分類 |
| **Coref** | Coreference Resolution — antecedent ランキング + クラスタ化 |

各タスクは独立して有効/無効を切り替えられます（`use_ner`, `use_rel`, `use_coref`）。

---

## 必要環境・インストール

### 動作確認環境

```
Python  >= 3.9
PyTorch >= 2.0
transformers >= 4.30
scipy >= 1.9
```

### インストール

```bash
pip install torch transformers scipy
```

GPU 使用（CUDA 11.8 の例）:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install transformers scipy
```

---

## ディレクトリ構成

```
dygie_standalone/
├── dygie/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # DyGIE++ JSON Lines データセット
│   │   └── collate.py        # バッチ化ユーティリティ
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dygie.py          # メインモデル (save/load 対応)
│   │   ├── span_extractor.py # スパン表現抽出（attentive pooling）
│   │   ├── ner_module.py     # NER ヘッド
│   │   ├── rel_module.py     # Relation ヘッド
│   │   └── coref_module.py   # Coref ヘッド
│   └── training/
│       ├── __init__.py
│       ├── trainer.py        # 学習ループ（AMP / Early stopping / Grad accum）
│       └── metrics.py        # NER / RE / CoNLL Coref F1
├── configs/
│   └── scierc.json           # SciERC 用設定例
├── scripts/
│   ├── train.py              # 学習スクリプト
│   ├── predict.py            # 推論スクリプト
│   └── evaluate.py           # 評価スクリプト
├── tests/
│   └── test_smoke.py         # スモークテスト（9 件）
└── data/sample/
    └── sample.jsonl          # 動作確認用サンプルデータ（3 文書）
```

---

## データ形式

DyGIE++ JSON Lines 形式（`.jsonl`）を使用します。  
**1 行 = 1 文書**。すべてのインデックスはドキュメント全体でのフラットなトークンインデックスです。

```jsonc
{
  "doc_key": "doc_001",           // 文書 ID（任意の文字列）
  "dataset": "scierc",            // データセット名（任意）
  "sentences": [                  // 文のリスト（必須）
    ["We", "present", "a", "neural", "model", "."],
    ["It", "uses", "BERT", "."]
  ],
  "ner": [                        // 文ごとの NER アノテーション（optional）
    [[3, 4, "Method"]],           // [start, end, label]  end は inclusive
    [[2, 2, "Method"]]
  ],
  "relations": [                  // 文ごとの RE アノテーション（optional）
    [[3, 4, 8, 10, "Used-for"]], // [s1, e1, s2, e2, label]
    []
  ],
  "clusters": [                   // Coref クラスタ（optional）
    [[3, 4], [7, 7]],             // 各クラスタは mention のリスト [start, end]
    [[2, 2], [13, 13]]
  ]
}
```

> **注意**: `ner` / `relations` / `clusters` はすべて省略可能です。  
> タスクを有効にしているのにフィールドが存在しない場合は、ラベルなし（すべて 0）として扱われます。

### サンプルデータ

```bash
cat data/sample/sample.jsonl
```

---

## クイックスタート

サンプルデータで動作確認を行います（GPU 不要）。

```bash
cd dygie_standalone

# 学習（サンプルデータで 2 エポックだけ走らせる例）
python scripts/train.py \
  --config configs/scierc.json \
  --train_path data/sample/sample.jsonl \
  --dev_path   data/sample/sample.jsonl \
  --output_dir output/quickstart \
  --transformer_model bert-base-cased \
  --num_epochs 2 \
  --batch_size 1

# 推論
python scripts/predict.py \
  --model_dir   output/quickstart/checkpoint_best \
  --input_path  data/sample/sample.jsonl \
  --output_path output/quickstart/predictions.jsonl

# 評価
python scripts/evaluate.py \
  --gold_path data/sample/sample.jsonl \
  --pred_path output/quickstart/predictions.jsonl
```

---

## 学習

### 基本コマンド

```bash
python scripts/train.py \
  --config     configs/scierc.json \
  --train_path data/train.jsonl \
  --dev_path   data/dev.jsonl \
  --output_dir output/scierc_bert
```

設定ファイル (`configs/scierc.json`) に書かれた値が使われます。  
CLI オプションで個別に上書きすることもできます。

### 主なオプション

| オプション | 型 | 説明 | デフォルト |
|---|---|---|---|
| `--config` | str | JSON 設定ファイルのパス（必須） | — |
| `--train_path` | str | 学習データ `.jsonl`（必須） | — |
| `--dev_path` | str | 検証データ `.jsonl`（必須） | — |
| `--output_dir` | str | 出力ディレクトリ（必須） | — |
| `--transformer_model` | str | HuggingFace モデル名またはローカルパス | 設定ファイルの値 |
| `--num_epochs` | int | 学習エポック数 | 20 |
| `--batch_size` | int | バッチサイズ | 4 |
| `--lr_transformer` | float | Transformer エンコーダの学習率 | 1e-5 |
| `--lr_task` | float | タスクヘッドの学習率 | 1e-4 |
| `--device` | str | `cuda` または `cpu`（省略で自動） | 自動 |
| `--use_amp` | flag | AMP（自動混合精度）を有効化 ※CUDA のみ | false |
| `--patience` | int | Early stopping の待機エポック数（0=無効） | 0 |
| `--gradient_accumulation_steps` | int | 勾配蓄積ステップ数 | 1 |

### GPU メモリが少ない場合の推奨設定

```bash
python scripts/train.py \
  --config configs/scierc.json \
  --train_path data/train.jsonl \
  --dev_path   data/dev.jsonl \
  --output_dir output/scierc_bert \
  --use_amp \
  --patience 5 \
  --gradient_accumulation_steps 4
```

- `--use_amp` : FP16 混合精度で VRAM を約半分に削減
- `--gradient_accumulation_steps 4` : バッチサイズ 4 を保ちながら実効バッチサイズ 16 に
- `--patience 5` : 5 エポック改善しなければ自動停止

### 設定ファイル (`configs/scierc.json`)

```jsonc
{
  "transformer_model": "allenai/scibert_scivocab_cased",
  "max_span_width": 8,           // 列挙するスパンの最大幅（トークン数）
  "max_total_length": 512,       // Transformer への最大入力長

  "use_ner":   true,
  "use_rel":   true,
  "use_coref": true,

  "ner_loss_weight":   1.0,      // 各タスク損失の重み
  "rel_loss_weight":   1.0,
  "coref_loss_weight": 1.0,
  "use_gold_spans_for_rel": true,// 学習時に gold NER スパンで RE を計算

  "num_epochs": 20,
  "batch_size": 4,
  "lr_transformer": 1e-5,
  "lr_task": 1e-4,
  "warmup_steps": 200,
  "max_grad_norm": 1.0,

  "use_amp": false,
  "patience": 5,
  "gradient_accumulation_steps": 1
}
```

### 出力ファイル

学習後、`output_dir` に以下が生成されます。

```
output/scierc_bert/
├── checkpoint_best/        # 最良モデル（dev スコア最高時）
│   ├── model.pt            # PyTorch state dict
│   ├── config.json         # Transformer encoder config（HuggingFace 形式）
│   ├── dygie_config.json   # DyGIE 固有設定（ラベル・ハイパーパラメータ）
│   └── labels.json         # NER / RE ラベル一覧（後方互換用）
├── checkpoint_last/        # 最終エポックのモデル
├── config.json             # 使用した学習設定
├── labels.json             # ラベル一覧
└── history.json            # エポックごとの loss / metrics 履歴
```

> `checkpoint_best` が推論・評価に使用します。

---

## 推論

```bash
python scripts/predict.py \
  --model_dir   output/scierc_bert/checkpoint_best \
  --input_path  data/test.jsonl \
  --output_path predictions.jsonl
```

### オプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--model_dir` | チェックポイントディレクトリ（必須） | — |
| `--input_path` | 入力 `.jsonl`（必須） | — |
| `--output_path` | 出力先ファイルパス（必須） | — |
| `--batch_size` | バッチサイズ | 4 |
| `--device` | `cuda` または `cpu` | 自動 |

### 出力形式

入力と同じ JSON Lines 形式で、予測結果がフィールドとして追加されます。

```jsonc
{
  "doc_key": "doc_001",
  "dataset": "scierc",
  "sentences": [...],
  "predicted_ner": [              // 文ごとの NER 予測
    [[3, 4, "Method"], [8, 10, "Task"]],
    [[2, 2, "Method"]]
  ],
  "predicted_relations": [        // 文ごとの RE 予測
    [[3, 4, 8, 10, "Used-for"]],
    []
  ],
  "predicted_clusters": [         // Coref クラスタ予測
    [[3, 4], [7, 7]]
  ]
}
```

---

## 評価

### 専用評価スクリプト

```bash
python scripts/evaluate.py \
  --gold_path data/test.jsonl \
  --pred_path predictions.jsonl
```

出力例:

```
===== Evaluation Results =====
NER  | P=0.7123  R=0.6891  F1=0.7005  (TP=423 FP=171 FN=191)
RE   | P=0.6432  R=0.5987  F1=0.6201  (TP=187 FP=104 FN=125)
Coref MUC   | P=0.8012  R=0.7654  F1=0.7829
Coref B³    | P=0.7234  R=0.6921  F1=0.7074
Coref CEAFφ4| P=0.6543  R=0.6789  F1=0.6664
CoNLL Score | F1=0.7189
==============================
```

### 指標の説明

| 指標 | 説明 |
|------|------|
| NER F1 | スパン（start, end, label）の完全一致 micro F1 |
| RE F1  | スパンペア（s1, e1, s2, e2, label）の完全一致 micro F1 |
| CoNLL Score | MUC / B³ / CEAFφ4 の平均 F1（CoNLL-2012 公式スコア） |

---

## モデルの保存と復元

### 保存

```python
model.save_pretrained("output/scierc_bert/checkpoint_best")
```

以下の 3 ファイルが保存されます:

| ファイル | 内容 |
|---|---|
| `model.pt` | 全パラメータの state dict（エンコーダ含む） |
| `config.json` | Transformer エンコーダ設定（HuggingFace 形式） |
| `dygie_config.json` | NER / RE ラベル・ハイパーパラメータ一式 |

### 復元

```python
from dygie.model import DyGIE

model = DyGIE.from_pretrained("output/scierc_bert/checkpoint_best")
```

`dygie_config.json` があれば**引数不要**で完全復元できます。  
個別パラメータを kwargs で上書きすることも可能です：

```python
model = DyGIE.from_pretrained(
    "output/scierc_bert/checkpoint_best",
    dropout=0.1,   # dropout だけ変更して復元
)
```

---

## Python API

### モデルの直接利用

```python
import torch
from transformers import AutoTokenizer
from dygie.model import DyGIE
from dygie.data import DyGIEDataset, collate_fn
from torch.utils.data import DataLoader

# モデル初期化
model = DyGIE(
    transformer_model="bert-base-cased",
    ner_labels=["Method", "Task", "Metric", "Material"],
    rel_labels=["Used-for", "Hyponym-of", "Compare"],
    max_span_width=8,
    use_ner=True,
    use_rel=True,
    use_coref=True,
)

# データロード
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
dataset = DyGIEDataset(
    path="data/train.jsonl",
    tokenizer=tokenizer,
    max_span_width=8,
)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# フォワードパス
batch = next(iter(loader))
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_to_subword=batch["token_to_subword"],
    spans=batch["spans"],
    span_mask=batch["span_mask"],
    num_tokens=batch["num_tokens"],
    ner_labels=batch["ner_labels"],    # 学習時のみ指定
    rel_labels=batch["rel_labels"],    # 学習時のみ指定
    coref_clusters=batch["coref_clusters"],  # 学習時のみ指定
)

print(outputs["loss"])        # 総損失
print(outputs["ner_preds"])   # [B, K] — スパンごとの NER 予測クラス ID
print(outputs["rel_preds"])   # [B, K, K] — スパンペアの RE 予測クラス ID
```

### 推論のみ（ラベルなし）

```python
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_to_subword=batch["token_to_subword"],
        spans=batch["spans"],
        span_mask=batch["span_mask"],
        num_tokens=batch["num_tokens"],
        use_gold_spans=False,   # 推論時は predicted NER スパンで RE を計算
    )
    # outputs["loss"] は存在しない
```

### Trainer の直接利用

```python
from dygie.training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    dev_loader=dev_loader,
    output_dir="output/my_model",
    num_epochs=20,
    lr_transformer=1e-5,
    lr_task=1e-4,
    warmup_steps=200,
    use_amp=True,               # AMP（CUDA のみ有効）
    patience=5,                 # Early stopping
    gradient_accumulation_steps=4,
)
trainer.train()

# 評価のみ
metrics = trainer.evaluate(dev_loader)
print(metrics)  # {"ner_f1": 0.72, "rel_f1": 0.61, "conll_f1": 0.71, ...}
```

---

## テスト

スモークテストを実行します（外部データ不要、CPU で約 30 秒）。

```bash
cd dygie_standalone
python -m pytest tests/test_smoke.py -v
```

### テスト内容

| テスト名 | 検証内容 |
|---|---|
| `test_dataset_loading` | データセット読み込み・スパン列挙・ラベル収集 |
| `test_dataloader_collate` | バッチ化・パディング整合性 |
| `test_forward_with_labels` | フォワードパス（損失計算・勾配） |
| `test_forward_no_labels` | 推論モード（損失なし・`loss` キーの不在） |
| `test_metrics` | NER / RE / Coref F1 計算の正確性 |
| `test_coref_metrics_precision_recall_independence` | B³ P と R が独立して計算されることの確認 |
| `test_save_load_pretrained` | save/load の往復（重み・設定の完全一致） |
| `test_backward_pass` | 損失から全パラメータへの勾配伝播 |
| `test_coref_distance_embedding` | Coref 距離埋め込みの方向性（causal masking） |

### 期待される出力

```
tests/test_smoke.py::test_dataset_loading                         PASSED
tests/test_smoke.py::test_dataloader_collate                      PASSED
tests/test_smoke.py::test_forward_with_labels                     PASSED
tests/test_smoke.py::test_forward_no_labels                       PASSED
tests/test_smoke.py::test_metrics                                  PASSED
tests/test_smoke.py::test_coref_metrics_precision_recall_independence PASSED
tests/test_smoke.py::test_save_load_pretrained                    PASSED
tests/test_smoke.py::test_backward_pass                           PASSED
tests/test_smoke.py::test_coref_distance_embedding                PASSED
========================= 9 passed in XX.XXs ==========================
```

---

## ハイパーパラメータ一覧

### モデルパラメータ

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `transformer_model` | HuggingFace モデル名またはローカルパス | — |
| `max_span_width` | 列挙するスパンの最大幅（トークン数） | 8 |
| `use_ner` | NER タスクを有効化 | true |
| `use_rel` | RE タスクを有効化 | true |
| `use_coref` | Coref タスクを有効化 | true |
| `ner_loss_weight` | NER 損失の重み | 1.0 |
| `rel_loss_weight` | RE 損失の重み | 1.0 |
| `coref_loss_weight` | Coref 損失の重み | 1.0 |
| `width_embedding_dim` | スパン幅埋め込みの次元数 | 128 |
| `feedforward_dim` | タスクヘッド FF 層の次元数 | 150 |
| `use_attentive_pooling` | attentive pooling でスパン表現を抽出 | true |
| `spans_per_word` | Coref mention pruning の割合 | 0.4 |
| `max_top_antecedents` | 各 mention に対して考慮する antecedent の最大数 | 50 |
| `dropout` | ドロップアウト率 | 0.4 |

### 学習パラメータ

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `num_epochs` | 最大エポック数 | 20 |
| `batch_size` | バッチサイズ | 4 |
| `lr_transformer` | Transformer エンコーダの学習率 | 1e-5 |
| `lr_task` | タスクヘッドの学習率 | 1e-4 |
| `warmup_steps` | 線形ウォームアップのステップ数 | 200 |
| `max_grad_norm` | 勾配クリッピングの閾値 | 1.0 |
| `use_gold_spans_for_rel` | 学習時 RE に gold NER スパンを使用 | true |
| `use_amp` | AMP（自動混合精度）※ CUDA のみ | false |
| `patience` | Early stopping の待機エポック数（0=無効） | 0 |
| `gradient_accumulation_steps` | 勾配蓄積ステップ数 | 1 |

---

## バグ修正と改善点 (v2)

### バグ修正

**1. Coref metrics recall の計算誤り** (`metrics.py`)

`_muc` / `_b3` / `_ceaf_phi4` の集計で、recall 分子に precision スコアを使っていたバグを修正しました。

```python
# Before (誤り): recall の計算に precision を使用
def _f1(triples):
    p = sum(t[0] for t in triples) / n   # precision
    # t[2] は recall だが t[0]+t[1] で間違った計算になっていた

# After (修正): (precision, recall) を正しく独立して計算
def _f1(pairs: list[tuple[float, float]]) -> dict[str, float]:
    p = sum(t[0] for t in pairs) / n
    r = sum(t[1] for t in pairs) / n
    f = 2 * p * r / (p + r + 1e-9)
```

**2. Coref antecedent 距離埋め込みの方向が逆** (`coref_module.py`)

有効な antecedent ペア（`j < i`）で距離が常に 0 になっていたバグを修正しました。

```python
# Before (誤り): dist[i,j] = max(0, j - i)  → j < i で常に 0
dist = (torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1)).clamp(min=0)

# After (修正): dist[i,j] = max(0, i - j)  → 正しい距離
dist = (torch.arange(T).unsqueeze(1) - torch.arange(T).unsqueeze(0)).clamp(min=0)
```

**3. `max_top_antecedents` が未適用** (`coref_module.py`)

`__init__` で受け取るが `_score_antecedents` で使われていなかった問題を修正しました。

```python
# After (修正): 距離ウィンドウとして適用
if self.max_top_antecedents < T - 1:
    too_far = dist > self.max_top_antecedents
    pair_scores = pair_scores.masked_fill(too_far.unsqueeze(0), float("-inf"))
```

**4. モデル保存時に DyGIE 設定が失われる** (`dygie.py`)

`save_pretrained` が `dygie_config.json`（NER/RE ラベル・ハイパーパラメータ）を保存するように対応し、`from_pretrained` が自動で復元するようにしました。

**5. デッドコード `tok_offset`** (`dataset.py`)

使われていない変数 `tok_offset` を削除しました。

**6. `torch.load` のセキュリティ警告** (`dygie.py`, `predict.py`)

PyTorch 2.0 以降のセキュリティ警告を抑制するため `weights_only=True` を追加しました。

### 機能追加

**7. AMP（自動混合精度）学習** (`trainer.py`)

```python
trainer = Trainer(..., use_amp=True)
```

CUDA 環境で FP16 混合精度学習を行い、速度向上と VRAM 削減を実現します。

**8. Early stopping** (`trainer.py`)

```python
trainer = Trainer(..., patience=5)
```

dev スコアが `patience` エポック連続で改善しない場合に学習を自動停止します。

**9. Gradient accumulation** (`trainer.py`)

```python
trainer = Trainer(..., gradient_accumulation_steps=4)
```

`gradient_accumulation_steps` ステップごとにパラメータを更新し、小さいバッチサイズでも実効バッチサイズを増やせます。

---

## トラブルシューティング

### `OSError: Error no file named model.safetensors...`

`from_pretrained` でモデルパスにスペルミスがある場合、または HuggingFace 形式でないディレクトリを指定した場合に発生します。  
`checkpoint_best` ディレクトリ（`dygie_config.json` と `model.pt` が入っているディレクトリ）を指定してください。

```bash
# 正しい例
python scripts/predict.py --model_dir output/scierc_bert/checkpoint_best ...
# 誤り（output_dir を直接指定）
python scripts/predict.py --model_dir output/scierc_bert ...
```

### `CUDA out of memory`

- `--batch_size` を 1–2 に減らす
- `--use_amp` を追加する
- `--gradient_accumulation_steps 4` などで実効バッチサイズを保ちながらメモリ使用を削減する
- `--max_span_width` を小さくする（8 → 5）

### `AMP requested but device is not CUDA. AMP disabled.`

`--use_amp` を指定しても CPU 環境では自動的に無効になります（エラーではありません）。

### ラベルが自動収集されない

`DyGIEDataset` に `ner_labels` / `rel_labels` を明示指定することで固定できます。  
訓練データと推論データで同じラベルセットを使うようにしてください（`train_ds.ner_labels` を `dev_ds` に渡す）。

### 学習が収束しない

- `lr_transformer` を下げる（1e-5 → 5e-6）
- `warmup_steps` を増やす
- `max_span_width` を小さくすることで候補スパン数を削減しノイズを減らす
