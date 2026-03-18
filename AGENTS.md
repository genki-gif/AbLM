# AbLM プロジェクト — エージェント向けリファレンス

## プロジェクト概要

OAS（Observed Antibody Space）からヒトペア抗体配列を取得し、3 種類の抗体言語モデル（AntiBERTy・AbLang・AbLang-2）で埋め込みベクトルを生成して UMAP 可視化するパイプライン。
- **V 遺伝子ファミリー（IGHV1〜IGHV7）** による色分けで、モデルごとの埋め込み空間の構造差を比較することが目的。
- AbLang-2 については **VH+VL ペア入力** と **VH 単独入力** の 2 条件を比較する。
- 追加解析: OLGA による VDJ 生成確率 (Pgen) と各モデルの疑似対数尤度 (PLL) の相関を可視化する。

---

## ディレクトリ構成

```
AbLM/
├── AGENTS.md                            # このファイル
├── environment.yml                      # mamba 仮想環境定義（ablm）
├── notebook.ipynb                       # UMAP + Pgen-PLL 可視化ノートブック
├── scripts/
│   ├── 01_download_data.py              # OAS ダウンロード・前処理・CDR3 特定
│   ├── 02_generate_embeddings.py        # 3モデル × 2条件の埋め込み生成
│   └── 03_compute_pgen_pll.py           # OLGA Pgen + 各モデル PLL 計算
├── data/
│   ├── sequences_processed.csv          # 前処理済み配列（8626 行 × 202 列）
│   ├── antiberty_seq_vecs.npy           # (8626, 512) float32
│   ├── antiberty_cdr3_vecs.npy          # (8626, 512) float32
│   ├── ablang_seq_vecs.npy              # (8626, 768) float32
│   ├── ablang_cdr3_vecs.npy             # (8626, 768) float32
│   ├── ablang2_seq_vecs.npy             # (8626, 480) float32  ← VH+VL ペア入力
│   ├── ablang2_cdr3_vecs.npy            # (8626, 480) float32  ← VH+VL ペア入力
│   ├── ablang2_heavy_seq_vecs.npy       # (8626, 480) float32  ← VH 単独入力
│   ├── ablang2_heavy_cdr3_vecs.npy      # (8626, 480) float32  ← VH 単独入力
│   ├── pgen.npy                         # (8626,) float64  OLGA Pgen (条件なし)
│   ├── pgen_vj.npy                      # (8626,) float64  OLGA Pgen (V/J 条件付き)
│   ├── antiberty_pll.npy                # (8626,) float32
│   ├── antiberty_cdr3_pll.npy           # (8626,) float32
│   ├── ablang_pll.npy                   # (8626,) float32
│   ├── ablang_cdr3_pll.npy              # (8626,) float32
│   ├── ablang2_paired_pll.npy           # (8626,) float32
│   ├── ablang2_paired_cdr3_pll.npy      # (8626,) float32
│   ├── ablang2_heavy_pll.npy            # (8626,) float32
│   ├── ablang2_heavy_cdr3_pll.npy       # (8626,) float32
│   └── pgen_pll_scores.csv              # 全スコア統合 CSV
└── figures/
    ├── umap_all_models.{png,pdf}            # 2×4 グリッド（全モデル比較）
    ├── umap_sequence_level.{png,pdf}        # シーケンスレベル 1×4
    ├── umap_cdr3_level.{png,pdf}            # CDR-H3 レベル 1×4
    ├── pgen_vs_cdr3_pll.{png,pdf}           # CDR3-PLL vs Pgen 1×4（主要比較）
    ├── pgen_vs_fullseq_pll.{png,pdf}        # Full-seq PLL vs Pgen 1×4（補足）
    └── pgen_vs_pll_all.{png,pdf}            # 統合図 2×4
```

---

## 仮想環境

```bash
# 環境構築（初回のみ）
mamba env create -f environment.yml

# スクリプト実行
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/<script.py>

# Jupyter カーネル（notebook.ipynb 実行時）
# カーネル名: "Python (ablm)"
# KMP_DUPLICATE_LIB_OK=TRUE は kernel.json に設定済み
# → ~/Library/Jupyter/kernels/ablm/kernel.json
```

**重要な注意点：**
- macOS では `torch` と OpenMP が競合するため `KMP_DUPLICATE_LIB_OK=TRUE` が必須
- `transformers` は `4.46.3` に固定（`antiberty 0.1.3` は v5 と非互換）

---

## データパイプライン

### Step 1: データ取得・前処理

```bash
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/01_download_data.py \
    --n 10000 [--force] [--local] [--outdir data]
```

**処理内容：**
1. `https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/` を巡回し、ヒト種のスタディのみ `.csv.gz` をダウンロード
2. 内部では `n * 5` 件の raw 配列を取得してから前処理でサンプリング
3. フィルタリング条件：
   - VH・VL 長さ ≥ 80aa
   - 非標準アミノ酸（`X * - .`）を含まない
   - V 遺伝子ファミリーが IGHV1〜IGHV7
4. V ファミリーごとに均等サンプリング（`n // n_families` 件）
5. CDR-H3 位置の特定：OAS の `cdr3_aa_heavy` カラムから直接取得、失敗時は `abnumber` にフォールバック
6. 出力: `data/sequences_processed.csv`

**実際の取得結果（n=10000 指定時）：**

| V ファミリー | 件数 |
|---|---|
| IGHV1 | 1428 |
| IGHV2 | 1428 |
| IGHV3 | 1428 |
| IGHV4 | 1428 |
| IGHV5 | 1428 |
| IGHV6 | 637（OAS に少ない） |
| IGHV7 | 849（OAS に少ない） |
| **合計** | **8626** |

**重要な CSV カラム：**

| カラム | 内容 |
|---|---|
| `vh_seq` | 重鎖 VH アミノ酸配列 |
| `vl_seq` | 軽鎖 VL アミノ酸配列 |
| `v_call_heavy` | V 遺伝子アレル名（例: `IGHV1-2*02`） |
| `v_family` | V 遺伝子ファミリー（例: `IGHV1`） |
| `cdr3_aa_heavy` | OAS の CDR-H3 アミノ酸配列 |
| `cdr3_indices` | VH 配列中の CDR-H3 位置 `(start, end)` |
| `cdr3_seq_extracted` | 実際に抽出された CDR-H3 配列 |

### Step 2: 埋め込み生成

```bash
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/02_generate_embeddings.py \
    [--model all|antiberty|ablang|ablang2|ablang2heavy] \
    [--batch 32] [--device auto|cpu|cuda] [--datadir data]
```

**キャッシュ機能あり**：`.npy` ファイルが既存ならスキップ。再生成には手動削除が必要。

### Step 3: Pgen + PLL 計算

```bash
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py \
    [--model all|pgen|antiberty|ablang|ablang2|ablang2heavy] \
    [--batch 32] [--device auto|cpu|cuda] [--datadir data]
```

**処理内容：**
1. OLGA (humanIGH モデル) で各配列の VDJ 生成確率を計算
   - `pgen`: V/J 条件なし
   - `pgen_vj`: V/J アレル条件付き（`v_call_heavy`, `j_call_heavy` を使用）
2. 各言語モデルの Pseudo-Log-Likelihood (PLL) を計算
   - **Full-seq PLL**: VH 全残基を 1 つずつマスクして平均
   - **CDR3-specific PLL**: VH 全体をモデルに入力しつつ CDR3 残基のみを対象に平均
3. `data/pgen_pll_scores.csv` に全スコアをまとめて保存

**OLGA CDR3 フォーマット変換：**
OAS の `cdr3_aa_heavy` は保存残基 C・W を含まない。OLGA の定義は C (inclusive) → W/F (inclusive) なので、
`cdr3_olga = 'C' + cdr3_aa_heavy + vh_seq[cdr3_end]` と変換している。

**推定計算時間（CPU、8626 配列）：**
- Pgen: numba (FastPgen) 使用で数分
- AntiBERTy PLL: 全配列 2-4h、CDR3 のみ 15-30min
- AbLang PLL: 全配列 2-4h、CDR3 のみ 15-30min
- AbLang-2 PLL: 全配列 4-8h（長い入力のため）、CDR3 のみ 30-60min

**キャッシュ機能あり**：モデルごとの `.npy` が既存ならスキップ。

---

## モデル別の実装詳細

### AntiBERTy

```python
from antiberty import AntiBERTyRunner
runner = AntiBERTyRunner()
embs = runner.embed([vh_seq])  # list of Tensor (L+2, 512)
# index 0 = [CLS], -1 = [SEP] → 除去して (L, 512) を使用
res_emb = emb[1:-1].detach().cpu().float().numpy()
```

- モデル重み: `~/miniforge3/envs/ablm/lib/.../antiberty/trained_models/AntiBERTy_md_smooth`
- 埋め込み次元: **512**

### AbLang

```python
import ablang
model = ablang.pretrained("heavy")
model.freeze()
res_out = model([vh_seq], mode="rescoding")  # list of (L, 768)
```

- モデル重み: `~/miniforge3/envs/ablm/lib/.../ablang/model-weights-heavy/amodel.pt`（327MB）
- 初回起動時に自動ダウンロード。過去に手動ダウンロードして解凍済み。
- 埋め込み次元: **768**

### AbLang-2（重要：トークン形式）

```python
import ablang2
model = ablang2.pretrained("ablang2-paired", device="cpu")
```

モデル重み: `~/miniforge3/envs/ablm/lib/.../ablang2/model-weights-ablang2-paired/model.pt`（171MB）

**トークン形式（`w_extra_tkns=True` を使用）：**

```
ペア入力:  [<, VH[0], ..., VH[-1], >, |, <, VL[0], ..., VL[-1], >]
              0   1         L_vh  L_vh+1  ...
VH単独:    [<, VH[0], ..., VH[-1], >, |]
              0   1         L_vh  L_vh+1
```

VH 残基は **必ず index 1 〜 1+len(VH)** に位置する。

```python
# ペア入力
pairs = [[vh, vl] for vh, vl in zip(vh_batch, vl_batch)]
tokenized = model.tokenizer(pairs, pad=True, w_extra_tkns=True, device=device)

# VH 単独入力（VL を空文字列にすることで同じトークン構造を維持）
pairs = [[vh, ""] for vh in vh_batch]
tokenized = model.tokenizer(pairs, pad=True, w_extra_tkns=True, device=device)

# 推論・VH 残基抽出（共通）
rescoding = model.AbRep(tokenized).last_hidden_states  # (B, L_padded, 480)
vh_emb = rescoding[j, 1:1 + len(vh), :]
```

埋め込み次元: **480**

**過去のバグ修正記録：**
旧コードは `w_extra_tkns=False` + `f"{vh}|{vl}"` 形式を使い、`rescoding[j, 1:1+l_vh]` で抽出していたが、この形式では CLS トークンが存在しないため VH の先頭残基が欠落し、代わりに `|` セパレータを含む誤った抽出になっていた。現在のコードは `w_extra_tkns=True` に修正済み。

---

## 埋め込みの意味

すべてのモデルで **全長配列をモデルに入力し、その出力から必要な残基ベクトルを抽出**している。CDR-H3 配列単独をモデルへ入力することはしていない。

| 出力ファイル | 内容 |
|---|---|
| `*_seq_vecs.npy` | 全残基の mean-pooling → (N, D) |
| `*_cdr3_vecs.npy` | CDR-H3 部分の残基ベクトルの mean-pooling → (N, D) |

CDR-H3 の残基インデックスは `cdr3_indices` カラムの `(start, end)` で指定。無効な場合は全残基平均にフォールバック。

---

## UMAP + Pgen-PLL 可視化（notebook.ipynb）

**実行方法：** カーネル「Python (ablm)」を選択して全セル実行

**セル構成：**
1. セットアップ（ライブラリ読み込み・ファイル存在確認）
2. データと埋め込みベクトルの読み込み
3. 色設定（Okabe-Ito 7色パレット）
4. UMAP 計算（8パネル分: 4モデル × 2レベル）
5. 2×4 サブプロット保存（`figures/umap_all_models.{png,pdf}`）
6. 個別図保存（`figures/umap_sequence_level.*`, `figures/umap_cdr3_level.*`）
7. [Pgen-PLL] セクションヘッダー
8. [Pgen-PLL] `pgen_pll_scores.csv` 読み込み・色設定
9. [Pgen-PLL] ヘルパー関数 (`plot_pgen_pll_panel`)
10. [Pgen-PLL] 図 1: CDR3-specific PLL vs log10(Pgen_VJ)（1×4）→ `pgen_vs_cdr3_pll.*`
11. [Pgen-PLL] 図 2: Full-seq PLL vs log10(Pgen_VJ)（1×4）→ `pgen_vs_fullseq_pll.*`
12. [Pgen-PLL] 図 3: 統合図 2×4 → `pgen_vs_pll_all.*`

**UMAP パラメータ：** `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`

**Pgen-PLL 可視化で使う統計：** Spearman ρ（`scipy.stats.spearmanr`）、p 値を各パネルに表示

**可視化モデル一覧：**
```python
MODEL_NAMES  = ["AntiBERTy", "AbLang", "AbLang-2 (paired)", "AbLang-2 (heavy)"]
LEVELS       = ["Sequence", "CDR-H3"]
PLL_CDR3_COLS = [
    "antiberty_cdr3_pll", "ablang_cdr3_pll",
    "ablang2_paired_cdr3_pll", "ablang2_heavy_cdr3_pll",
]
```

---

## モデルの初回セットアップ状況

以下は過去に手動でモデル重みをダウンロード・解凍済み：

| モデル | 重みファイルの場所 | サイズ |
|---|---|---|
| AbLang (heavy) | `~/miniforge3/envs/ablm/lib/.../ablang/model-weights-heavy/amodel.pt` | 327MB |
| AbLang-2 (paired) | `~/miniforge3/envs/ablm/lib/.../ablang2/model-weights-ablang2-paired/model.pt` | 171MB |

AntiBERTy の重みはパッケージに同梱済み（別途ダウンロード不要）。

---

## よくあるエラーと対処

| エラー | 原因 | 対処 |
|---|---|---|
| `OMP: Error #15` | macOS での OpenMP 多重ロード | `KMP_DUPLICATE_LIB_OK=TRUE` を設定 |
| `AttributeError: 'AntiBERTy' object has no attribute 'all_tied_weights_keys'` | transformers v5 との非互換 | `pip install "transformers==4.46.3"` |
| OAS API 404 エラー | 旧 API エンドポイント (`/api/pairednumbers/`) が廃止 | 現在は `ngsdb/paired/` を直接巡回（実装済み） |
| `tmp.tar.gz` が 0B のまま停止 | AbLang/AbLang-2 の `requests.get()` によるダウンロードがハング | `curl` で手動ダウンロードして解凍（実施済み） |
| OLGA `Pgen = 0` が大量に出る | CDR3 の末端残基が W/F でない・V/J アレルが OLGA モデルに未収録 | 正常。`NaN` として記録され log10 変換時に自動除外される |

---

## 拡張・変更時の注意点

**配列数を変更する場合：**
```bash
# sequences_processed.csv と全 .npy を削除してから
rm data/sequences_processed.csv data/*.npy
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/01_download_data.py --n <新しい件数>
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/02_generate_embeddings.py
```

**Pgen/PLL スコアを再計算する場合：**
```bash
rm data/pgen.npy data/pgen_vj.npy data/*_pll.npy data/pgen_pll_scores.csv
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py
```

**新しいモデルを追加する場合（`02_generate_embeddings.py`）：**
1. `run_<model>()` 関数を追加（`pool_seq` / `pool_cdr3` ユーティリティ再利用可）
2. `main()` の `models_to_run` 分岐に追加
3. `--model` の help テキストを更新
4. `notebook.ipynb` の `vectors` 辞書と `MODEL_NAMES` リストを更新

**notebook.ipynb を直接編集する場合：**
セル内容は JSON 文字列として格納されているため、EditNotebook ツールではなく Python で直接 JSON を操作する方が確実：
```python
import json
with open('notebook.ipynb') as f:
    nb = json.load(f)
# nb['cells'][i]['source'] = "新しいコード"
with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```
