# DyGIE-- (DyGIE++ Standalone)

AllenNLP **不要**の DyGIE++ 再実装です。  
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
11. [LoRA ファインチューニング](#lora-ファインチューニング)
12. [Python API](#python-api)
13. [テスト](#テスト)
14. [ハイパーパラメータ一覧](#ハイパーパラメータ一覧)
15. [バージョン別変更履歴](#バージョン別変更履歴)
16. [トラブルシューティング](#トラブルシューティング)

---

## 概要

DyGIE++ (Wadden et al., 2019, EMNLP) は、Transformer エンコーダをバックボーンとする情報抽出モデルです。  
固有表現認識 (NER)・関係抽出 (RE)・共参照解析 (Coref)・イベント抽出 (Event) を統一的なスパンベースの枠組みで扱います。

本実装の特徴:

- **AllenNLP ゼロ依存** — PyTorch + Transformers だけで動作
- **SciERC / ACE 05 / DyGIE++ JSON Lines 形式**をそのまま入力可能
- **論文準拠のフォワード順序** — Coref → SpanPropagation → NER → Coref損失 → RE → Event
- **AMP・Early stopping・Gradient accumulation・Gradient checkpointing** をサポート
- **RE スコア改善** — タイプ埋め込み・距離特徴・Focal Loss
- **イベント抽出** — ACE 05 スタイルのトリガー検出 + 引数抽出
- **LoRA ファインチューニング** — 小規模データでの過学習抑制・GPU メモリ削減
- **`save_pretrained` / `from_pretrained`** による簡単なモデル保存・復元

---

## 対応タスク

| タスク | 概要 | 有効化フラグ |
|--------|------|------|
| **NER** | Named Entity Recognition — スパン分類 | `use_ner` |
| **RE**  | Relation Extraction — スパンペア分類 | `use_rel` |
| **Coref** | Coreference Resolution — antecedent ランキング + クラスタ化 | `use_coref` |
| **Event** | Event Extraction — トリガー検出 + 引数ロール分類 | `use_event` |

各タスクは独立して有効/無効を切り替えられます。

---

## 必要環境・インストール

### 動作確認環境

```
Python     >= 3.9
PyTorch    >= 2.0
transformers >= 4.30
scipy      >= 1.9
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

LoRA を使用する場合（オプション）:

```bash
pip install peft
```

---

## ディレクトリ構成

```
dygie_standalone/
├── dygie/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # DyGIE++ JSON Lines データセット（イベント対応）
│   │   └── collate.py          # バッチ化ユーティリティ
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dygie.py            # メインモデル（LoRA / save_load 対応）
│   │   ├── span_extractor.py   # スパン表現抽出（attentive pooling）
│   │   ├── span_propagation.py # スパングラフ伝播（Section 3.3）
│   │   ├── ner_module.py       # NER ヘッド
│   │   ├── rel_module.py       # Relation ヘッド（タイプ埋め込み・距離特徴・Focal Loss）
│   │   ├── coref_module.py     # Coref ヘッド（2 フェーズ構成）
│   │   └── event_module.py     # Event ヘッド（トリガー + 引数）
│   └── training/
│       ├── __init__.py
│       ├── trainer.py          # 学習ループ（AMP / Early stopping / Grad accum / LoRA対応）
│       └── metrics.py          # NER / RE / CoNLL Coref / Event F1
├── configs/
│   ├── scierc.json             # SciERC 用設定例（全オプション記載）
│   └── ace05_event.json        # ACE 05 イベント抽出用設定例
├── scripts/
│   ├── train.py                # 学習スクリプト
│   ├── predict.py              # 推論スクリプト
│   └── evaluate.py             # 評価スクリプト
├── tests/
│   └── test_smoke.py           # スモークテスト（14 件）
└── data/sample/
    ├── sample.jsonl            # NER / RE / Coref 動作確認用（3 文書）
    └── sample_event.jsonl      # Event 動作確認用
```

---

## データ形式

DyGIE++ JSON Lines 形式（`.jsonl`）を使用します。  
**1 行 = 1 文書**。すべてのインデックスはドキュメント全体でのフラットなトークンインデックスです。

### NER / RE / Coref

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

### イベント抽出 (ACE 05 スタイル)

```jsonc
{
  "doc_key": "doc_002",
  "dataset": "ace05",
  "sentences": [
    ["John", "died", "in", "Paris", "."],
    ["Mary", "was", "born", "in", "London", "."]
  ],
  "ner": [[[0, 0, "PER"], [3, 3, "GPE"]], [[5, 5, "PER"], [9, 9, "GPE"]]],
  "relations": [[], []],
  "events": [                     // 文ごとのイベントリスト（optional）
    [                             // sentence 0 のイベント
      [                           // 1 つのイベント = [ トリガー, 引数1, 引数2, ... ]
        [1, 1, "Life:Die"],       // トリガー: [start, end, event_type]
        [0, 0, "Person"]          // 引数:     [start, end, role]
      ]
    ],
    [                             // sentence 1 のイベント
      [[7, 7, "Life:Be-Born"], [5, 5, "Person"]]
    ]
  ]
}
```

> **注意**: 各フィールドは省略可能です。タスクを有効にしているのにフィールドが存在しない場合は、ラベルなし（すべて 0）として扱われます。

---

## クイックスタート

サンプルデータで動作確認を行います（GPU 不要）。

```bash
cd dygie_standalone

# NER / RE / Coref 学習（2 エポック）
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

### イベント抽出（ACE 05）

```bash
python scripts/train.py \
  --config     configs/ace05_event.json \
  --train_path data/ace05/train.jsonl \
  --dev_path   data/ace05/dev.jsonl \
  --output_dir output/ace05_event
```

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
| `--early_stopping_warmup` | int | Early stopping を開始するまでのウォームアップエポック数 | 20 |
| `--gradient_accumulation_steps` | int | 勾配蓄積ステップ数 | 1 |
| `--use_gradient_checkpointing` | flag | Transformer に勾配チェックポイントを適用（メモリ削減） | false |
| `--max_spans` | int | 文書あたりの最大スパン数（0=上限なし） | 0 |
| `--type_embedding_dim` | int | RE タイプ埋め込み次元数（0=無効） | 0 |
| `--use_distance_feature` | flag | RE スパン間距離特徴を有効化 | false |
| `--focal_loss_gamma` | float | RE Focal Loss の gamma 値（0=通常の CE） | 0.0 |
| `--use_event` | flag | イベント抽出タスクを有効化 | false |
| `--event_loss_weight` | float | イベント損失の重み | 1.0 |
| `--use_lora` | flag | LoRA ファインチューニングを有効化（peft 必要） | false |
| `--lora_r` | int | LoRA のランク | 8 |
| `--lora_alpha` | int | LoRA のスケーリング係数 | 32 |
| `--lora_dropout` | float | LoRA アダプタ内の Dropout 率 | 0.1 |

### GPU メモリが少ない場合の推奨設定

```bash
python scripts/train.py \
  --config configs/scierc.json \
  --train_path data/train.jsonl \
  --dev_path   data/dev.jsonl \
  --output_dir output/scierc_bert \
  --use_amp \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 4 \
  --patience 5 \
  --early_stopping_warmup 10
```

| オプション | 効果 |
|---|---|
| `--use_amp` | FP16 混合精度で VRAM を約 30〜50% 削減 |
| `--use_gradient_checkpointing` | 活性化メモリを 50〜70% 削減（速度は約 1.3x 低下） |
| `--gradient_accumulation_steps 4` | 実効バッチサイズ 16 相当を小 VRAM で実現 |
| `--patience 5` | 5 エポック改善なしで自動停止 |
| `--early_stopping_warmup 10` | 最初の 10 エポックは patience カウンタを増加させない |

### 設定ファイル (`configs/scierc.json`)

設定ファイルの主要フィールドを示します（全フィールドはファイル内のコメントを参照）。

```jsonc
{
  "transformer_model": "allenai/scibert_scivocab_cased",
  "max_span_width": 8,
  "max_total_length": 512,

  "use_ner": true, "use_rel": true, "use_coref": true,
  "ner_loss_weight": 1.0, "rel_loss_weight": 1.0, "coref_loss_weight": 1.0,
  "use_gold_spans_for_rel": true,

  "num_epochs": 20, "batch_size": 4,
  "lr_transformer": 1e-5, "lr_task": 1e-4,
  "warmup_steps": 200, "max_grad_norm": 1.0,

  // v2: 学習効率化
  "use_amp": false,
  "patience": 5,
  "early_stopping_warmup": 20,
  "gradient_accumulation_steps": 1,

  // v3: メモリ最適化
  "use_gradient_checkpointing": true,
  "max_spans": 0,

  // v4: RE スコア改善
  "type_embedding_dim": 128,
  "use_distance_feature": true,
  "num_distance_buckets": 10,
  "distance_embedding_dim": 64,
  "focal_loss_gamma": 2.0,

  // v6: LoRA（デフォルト無効）
  "use_lora": false,
  "lora_r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "lora_target_modules": ["query", "value"]
}
```

### 出力ファイル

学習後、`output_dir` に以下が生成されます。

```
output/scierc_bert/
├── checkpoint_best/        # 最良モデル（dev スコア最高時）
│   ├── model.pt            # PyTorch state dict（LoRA 重みを含む）
│   ├── config.json         # Transformer encoder config（HuggingFace 形式）
│   ├── dygie_config.json   # DyGIE 固有設定（ラベル・LoRA 設定・ハイパーパラメータ）
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
  "sentences": [...],
  "predicted_ner": [
    [[3, 4, "Method"], [8, 10, "Task"]],
    [[2, 2, "Method"]]
  ],
  "predicted_relations": [
    [[3, 4, 8, 10, "Used-for"]],
    []
  ],
  "predicted_clusters": [
    [[3, 4], [7, 7]]
  ],
  // イベント抽出が有効な場合
  "predicted_events": [
    [[[1, 1, "Life:Die"], [0, 0, "Person"]]],
    []
  ]
}
```

---

## 評価

```bash
python scripts/evaluate.py \
  --gold_path data/test.jsonl \
  --pred_path predictions.jsonl
```

出力例:

```
===== Evaluation Results =====
NER    | P=0.7123  R=0.6891  F1=0.7005
RE     | P=0.6432  R=0.5987  F1=0.6201
Coref  MUC    | P=0.8012  R=0.7654  F1=0.7829
Coref  B³     | P=0.7234  R=0.6921  F1=0.7074
Coref  CEAFφ4 | P=0.6543  R=0.6789  F1=0.6664
CoNLL Score   | F1=0.7189
Event Trigger | P=0.7500  R=0.7100  F1=0.7295
Event Arg     | P=0.6800  R=0.6300  F1=0.6540
==============================
```

---

## モデルの保存と復元

### 保存

```python
model.save_pretrained("output/scierc_bert/checkpoint_best")
```

以下の 3 ファイルが保存されます:

| ファイル | 内容 |
|---|---|
| `model.pt` | 全パラメータの state dict（LoRA 重みを含む） |
| `config.json` | Transformer エンコーダ設定（HuggingFace 形式） |
| `dygie_config.json` | NER / RE / Event ラベル・LoRA 設定・全ハイパーパラメータ |

### 復元

```python
from dygie.model import DyGIE

# dygie_config.json があれば引数不要で完全復元
model = DyGIE.from_pretrained("output/scierc_bert/checkpoint_best")
```

LoRA を有効にして学習したモデルは、`from_pretrained` で自動的に LoRA 構造が再構築されます。

```python
# 個別パラメータを上書きして復元することも可能
model = DyGIE.from_pretrained(
    "output/scierc_bert/checkpoint_best",
    dropout=0.1,
)
```

---

## LoRA ファインチューニング

LoRA (Low-Rank Adaptation, Hu et al., 2022) は BERT エンコーダの大部分を凍結し、  
低ランク行列 A, B のみを学習する手法です。

### メリット

| 項目 | 効果 |
|---|---|
| **GPU メモリ削減** | 学習対象パラメータが約 0.3〜1% になり、勾配・オプティマイザ状態が大幅減（SciBERT で Full FT 比 **約 75% 削減**） |
| **過学習抑制** | SciERC（500 文書程度）のような小規模データで有効 |
| **学習速度** | バックワードパスが軽くなり若干高速化 |

詳細な GPU メモリ比較（SciBERT、`batch_size=4`）:

| 設定 | GPU メモリ目安 |
|---|---|
| Full FT（grad ckpt なし） | 12〜16 GB |
| Full FT + grad ckpt | 7〜10 GB |
| **LoRA（grad ckpt なし）** | **5〜7 GB** |
| **LoRA + grad ckpt** | **4〜5 GB** |

### 使い方

**事前準備**:

```bash
pip install peft
```

**設定ファイルで有効化** (`configs/scierc.json`):

```json
{
  "use_lora": true,
  "lora_r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "lora_target_modules": ["query", "value"]
}
```

**CLI で有効化**:

```bash
python scripts/train.py \
  --config configs/scierc.json \
  --train_path data/train.jsonl \
  --dev_path   data/dev.jsonl \
  --output_dir output/scierc_lora \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 32
```

### LoRA パラメータの目安

| パラメータ | 説明 | 推奨値 |
|---|---|---|
| `lora_r` | LoRA のランク。低いほどパラメータが少ない | 4〜16 |
| `lora_alpha` | スケーリング係数。通常 `lora_r` の 2〜4 倍 | 8〜32 |
| `lora_dropout` | LoRA アダプタ内の Dropout 率 | 0.0〜0.1 |
| `lora_target_modules` | LoRA を適用する BERT のモジュール名 | `["query", "value"]` |

`lora_target_modules` を `["query", "key", "value", "dense"]` まで広げると表現力が上がりますが、パラメータ数も増えます。

### 学習可能パラメータ数の確認

```python
model = DyGIE(..., use_lora=True, lora_r=8)
model.print_trainable_parameters()
# → Trainable params: 295,168 / 109,483,778 (0.27%)
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
    transformer_model="allenai/scibert_scivocab_cased",
    ner_labels=["Method", "Task", "Metric", "Material"],
    rel_labels=["Used-for", "Hyponym-of", "Compare"],
    max_span_width=8,
    use_ner=True,
    use_rel=True,
    use_coref=True,
    # v4: RE 改善
    type_embedding_dim=128,
    use_distance_feature=True,
    focal_loss_gamma=2.0,
    # v6: LoRA（オプション）
    use_lora=True,
    lora_r=8,
    lora_alpha=32,
)

# データロード
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
dataset = DyGIEDataset(
    path="data/train.jsonl",
    tokenizer=tokenizer,
    max_span_width=8,
)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# フォワードパス（学習時）
batch = next(iter(loader))
outputs = model(
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
print(outputs["loss"])        # 総損失
print(outputs["ner_preds"])   # [B, K] スパンごとの NER 予測クラス ID
print(outputs["rel_preds"])   # [B, K, K] スパンペアの RE 予測クラス ID
```

### イベント抽出モデル

```python
model = DyGIE(
    transformer_model="bert-base-cased",
    ner_labels=["PER", "ORG", "GPE"],
    rel_labels=[],
    max_span_width=8,
    use_ner=True,
    use_rel=False,
    use_coref=False,
    use_event=True,
    event_type_labels=["Life:Die", "Life:Be-Born", "Business:Merge-Org"],
    arg_role_labels=["Person", "Place", "Org"],
    event_loss_weight=1.0,
)

outputs = model(
    ...,
    event_trigger_labels=batch["event_trigger_labels"],  # [B, K]
    event_arg_labels=batch["event_arg_labels"],          # [B, K, K]
)
print(outputs["event_loss"])           # イベント損失
print(outputs["event_trigger_preds"]) # [B, K] トリガー予測
print(outputs["event_arg_preds"])      # [B, K, K] 引数ロール予測
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
        use_gold_spans=False,  # 推論時は predicted NER スパンで RE を計算
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
    use_amp=True,                      # AMP（CUDA のみ有効）
    patience=5,                        # Early stopping
    early_stopping_warmup=10,          # 最初 10 エポックは停止しない
    gradient_accumulation_steps=4,
)
trainer.train()

# 評価のみ
metrics = trainer.evaluate(dev_loader)
# {"ner_f1": 0.72, "rel_f1": 0.61, "conll_f1": 0.71, ...}
```

---

## テスト

スモークテストを実行します（外部データ不要、CPU で約 30 秒）。

```bash
cd dygie_standalone
python tests/test_smoke.py
```

### テスト内容（14 件）

| テスト名 | 検証内容 |
|---|---|
| `Dataset` | データセット読み込み・スパン列挙・ラベル自動収集 |
| `DataLoader` | バッチ化・パディング整合性 |
| `Forward + Loss` | フォワードパス（NER/RE/Coref、損失計算） |
| `Inference (no labels)` | 推論モード（損失なし・`loss` キーの不在） |
| `Metrics` | NER / RE / CoNLL Coref F1 計算の正確性 |
| `Coref P/R independence` | B³ P と R が独立して計算されることの確認 |
| `Save / Load pretrained` | save/load の往復（重み・設定の完全一致） |
| `Backward pass` | 損失から全パラメータへの勾配伝播 |
| `RE v4 features` | タイプ埋め込み・距離特徴・Focal Loss の動作確認 |
| `Coref distance embedding` | Coref 距離埋め込みの方向性（causal masking） |
| `Event Metrics` | EventMetrics のトリガー / 引数 F1 計算の正確性 |
| `Event Forward + Loss` | イベント抽出モデルの forward / backward |
| `LoRA forward/backward` | LoRA 有効時の forward/backward・凍結パラメータの勾配なし確認 |
| `LoRA save / load pretrained` | LoRA モデルの save/load（パラメータ・出力の完全一致） |

LoRA テスト（最後の 2 件）は `peft` がインストールされていない場合はスキップされます。

### 期待される出力

```
===== DyGIE++ Standalone Smoke Tests =====
[Dataset]            [OK] Dataset: 3 docs, ...
[DataLoader]         [OK] DataLoader: input_ids=(2, 23), ...
...
[LoRA save / load pretrained]  [OK] LoRA save/load: use_lora=True | ...

===== Results: 14 passed / 0 failed =====
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
| `use_event` | Event タスクを有効化 | false |
| `ner_loss_weight` | NER 損失の重み | 1.0 |
| `rel_loss_weight` | RE 損失の重み | 1.0 |
| `coref_loss_weight` | Coref 損失の重み | 1.0 |
| `event_loss_weight` | Event 損失の重み | 1.0 |
| `width_embedding_dim` | スパン幅埋め込みの次元数 | 128 |
| `feedforward_dim` | タスクヘッド FF 層の次元数 | 150 |
| `use_attentive_pooling` | attentive pooling でスパン表現を抽出 | true |
| `spans_per_word` | Coref mention pruning の割合 | 0.4 |
| `max_top_antecedents` | 各 mention に対して考慮する antecedent の最大数 | 50 |
| `dropout` | ドロップアウト率 | 0.4 |
| `type_embedding_dim` | RE エンティティタイプ埋め込みの次元数（0=無効） | 0 |
| `use_distance_feature` | RE スパン間距離特徴を使用するか | false |
| `num_distance_buckets` | 距離バケット数（対数スケール） | 10 |
| `distance_embedding_dim` | 距離埋め込みの次元数（0=無効） | 64 |
| `focal_loss_gamma` | RE Focal Loss の gamma 値（0=通常の CE） | 0.0 |

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
| `early_stopping_warmup` | Early stopping を開始するまでのウォームアップエポック数 | 20 |
| `gradient_accumulation_steps` | 勾配蓄積ステップ数 | 1 |
| `use_gradient_checkpointing` | Transformer エンコーダに勾配チェックポイントを適用 | false |
| `max_spans` | 文書あたりの最大スパン数（0=上限なし） | 0 |

### LoRA パラメータ（v6）

| パラメータ | 説明 | デフォルト |
|---|---|---|
| `use_lora` | LoRA ファインチューニングを有効化（pip install peft 必要） | false |
| `lora_r` | LoRA のランク（推奨: 4〜16） | 8 |
| `lora_alpha` | LoRA のスケーリング係数（通常 lora_r の 2〜4 倍） | 32 |
| `lora_dropout` | LoRA アダプタ内の Dropout 率 | 0.1 |
| `lora_target_modules` | LoRA を適用する BERT のモジュール名 | `["query", "value"]` |

---

## バージョン別変更履歴

### v6 — LoRA ファインチューニング

- `dygie/model/dygie.py`: `use_lora` フラグで BERT エンコーダを LoRA 化（peft ライブラリ利用）
- `dygie/training/trainer.py`: Optimizer を `requires_grad=True` フィルタで構築（凍結 BERT 除外）
- `scripts/train.py`: `--use_lora / --lora_r / --lora_alpha / --lora_dropout` CLI 引数追加
- `configs/scierc.json`: LoRA 設定セクション追加
- `tests/test_smoke.py`: LoRA forward/backward・save/load テスト追加（14 件）

### v5 — イベント抽出 / スパングラフ伝播

- `dygie/model/event_module.py`: トリガー検出 + 引数ロール分類（O(T×K) 効率実装）
- `dygie/model/span_propagation.py`: DyGIE++ Section 3.3 スパングラフ伝播
- `dygie/model/dygie.py`: 論文準拠フォワード順序に変更
  - Coref `compute_representations` → SpanPropagation → NER → Coref `predict_labels` → RE → Event
- `dygie/model/coref_module.py`: 2 フェーズ構成に分割（`compute_representations` / `predict_labels`）
- `dygie/training/metrics.py`: `EventMetrics`（トリガー / 引数 F1）追加
- `configs/ace05_event.json`: ACE 05 イベント抽出用設定ファイル追加

### v4 — RE スコア改善

- `dygie/model/rel_module.py`:
  - エンティティタイプ埋め込みを RE ペア表現に追加（`type_embedding_dim`）
  - スパン間距離特徴（`use_distance_feature` / `distance_embedding_dim`）
  - Focal Loss による RE クラス不均衡への対処（`focal_loss_gamma`）
  - Pair MLP を 2 層 + LayerNorm に深化

### v3 — メモリ最適化

- `dygie.py`: `use_gradient_checkpointing` オプション追加
- `data/dataset.py`: `max_spans` による RE ペア行列の K 制限

### v2 — バグ修正・学習効率化

**バグ修正**:

1. **Coref metrics recall の計算誤り** — recall 分子に precision スコアを使っていたバグを修正
2. **Coref antecedent 距離埋め込みの方向が逆** — `dist[i,j] = max(0, j-i)` → `max(0, i-j)` に修正
3. **`max_top_antecedents` が未適用** — `_score_antecedents` で距離ウィンドウとして適用
4. **モデル保存時に DyGIE 設定が失われる** — `dygie_config.json` を保存・復元するように対応
5. **`patience=0` 時の OverflowError** — `%d` フォーマットで `float("inf")` を渡していたバグを修正
6. **デッドコード `tok_offset`** — 未使用変数を削除
7. **`torch.load` のセキュリティ警告** — `weights_only=True` を追加

**機能追加**:

8. **AMP（自動混合精度）学習** — `use_amp=True`
9. **Early stopping** — `patience` エポック連続で改善なしの場合に自動停止
10. **Early stopping ウォームアップ** — `early_stopping_warmup` エポック中は patience を増加しない
11. **Gradient accumulation** — `gradient_accumulation_steps` ステップごとにパラメータ更新

---

## トラブルシューティング

### `CUDA out of memory`

推奨順に試してください：

1. `--use_gradient_checkpointing` を追加（最も効果的、活性化メモリ 50〜70% 削減）
2. `--use_lora` を追加（勾配・オプティマイザ状態 約 75% 削減、`pip install peft` 必要）
3. `--batch_size` を 1〜2 に減らし、`--gradient_accumulation_steps 4` で実効バッチサイズを補う
4. `--use_amp` を追加（FP16 混合精度）
5. `--max_spans 800` で RE ペア行列の K を制限
6. `--max_span_width 5` でスパン候補数を削減

### `OSError: Error no file named model.safetensors...`

`from_pretrained` でモデルパスにスペルミスがある場合、または `checkpoint_best` ではなく `output_dir` を直接指定した場合に発生します。

```bash
# 正しい例（checkpoint_best を指定）
python scripts/predict.py --model_dir output/scierc_bert/checkpoint_best ...

# 誤り（output_dir を直接指定）
python scripts/predict.py --model_dir output/scierc_bert ...
```

### `AMP requested but device is not CUDA. AMP disabled.`

`--use_amp` を指定しても CPU 環境では自動的に無効になります（エラーではありません）。

### `ImportError: LoRA を使用するには peft ライブラリが必要です`

```bash
pip install peft
```

### ラベルが自動収集されない

`DyGIEDataset` に `ner_labels` / `rel_labels` を明示指定することで固定できます。  
訓練データと検証・テストデータで同じラベルセットを使うようにしてください。

```python
train_ds = DyGIEDataset(path="train.jsonl", ...)
dev_ds   = DyGIEDataset(
    path="dev.jsonl",
    ner_labels=train_ds.ner_labels,  # 学習データのラベルセットを共有
    rel_labels=train_ds.rel_labels,
    ...
)
```

### 学習が収束しない

- `lr_transformer` を下げる（1e-5 → 5e-6）
- `warmup_steps` を増やす（200 → 500）
- `max_span_width` を小さくして候補スパン数を削減
- `use_lora=True` で過学習を抑制（小規模データ）
- `focal_loss_gamma=2.0` で RE のクラス不均衡に対処

---

## 参考文献

- Wadden et al. (2019). *Entity, Relation, and Event Extraction with Contextualized Span Representations.* EMNLP. https://aclanthology.org/D19-1585/
- Lee et al. (2018). *Higher-Order Coreference Resolution with Coarse-to-Fine Inference.* NAACL. https://aclanthology.org/N18-2108/
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. https://arxiv.org/abs/2106.09685
