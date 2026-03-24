# CLAUDE.md

このファイルは Claude Code (claude.ai/code) がこのリポジトリで作業する際のガイドラインを提供する。

## プロジェクト概要

AbLM は、OAS (Observed Antibody Space) からヒトペア抗体配列をダウンロードし、複数の抗体言語モデル（AntiBERTy, AbLang, AbLang-2, ESM-2, ProGen-2）で埋め込みを生成し、UMAP 可視化・Pgen-PLL 相関分析・FLAb2 ベンチマーク分類評価を行うパイプラインである。研究目的は、各モデルが抗体埋め込み空間をどのように構造化するかを V 遺伝子ファミリー（IGHV1–7）着色で比較すること。

## 環境構築

```bash
# 環境作成（初回のみ）
mamba env create -f environment.yml

# スクリプト実行
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/<script.py>
```

**重要な制約:**
- `KMP_DUPLICATE_LIB_OK=TRUE` が必須（PyTorch/OpenMP の競合回避）
- `transformers` は `4.46.3` に固定 — AntiBERTy 0.1.3 は v5 以降と非互換

## パイプライン実行

```bash
# ステップ 1: OAS から配列をダウンロード・前処理（約 8626 配列）
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/01_download_data.py --n 10000

# ステップ 2: 全モデルの埋め込み生成
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/02_generate_embeddings.py --batch 32
# --model: all | antiberty | ablang | ablang2 | ablang2heavy | esm2 | progen2

# ステップ 3: Pgen（OLGA）と PLL スコアの計算
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py --batch 32
# --model: all | pgen | antiberty | ablang | ablang2 | ablang2heavy

# ステップ 4: FLAb2 ベンチマーク分類評価
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/04_benchmark_classification.py --batch 32
# --model: all | antiberty | ablang | ablang2heavy | esm2 | progen2
# --datasets: all | カンマ区切り ID

# ステップ 4.5: 治療用抗体データ取得 + Pgen/PLL スコア計算
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/05_therapeutic_data.py --batch 32
# --source: therasabdab | fallback
# --model: all | pgen | antiberty | ablang | ablang2 | ablang2heavy | esm2 | progen2

# ステップ 5: ノートブックで可視化
# カーネル "Python (ablm)" を選択して全セル実行
```

ステップ 2・3・4・4.5 は**キャッシュ機能**あり：`.npy` ファイルが存在すればスキップされる。再生成するには手動で削除する。

## アーキテクチャ

### データフロー

```
OAS API → 01_download_data.py → data/sequences_processed.csv (8626 × 202)
                                         ↓
                          02_generate_embeddings.py → 12 個の .npy 埋め込みファイル
                          [AntiBERTy(512d), AbLang(768d), AbLang-2 paired(480d),
                           AbLang-2 heavy(480d), ESM-2(1280d), ProGen-2(1024d)]
                          [× sequence レベル / CDR-H3 レベル プーリング]
                                         ↓
                          03_compute_pgen_pll.py → 10 個の .npy スコアファイル + pgen_pll_scores.csv
                          [OLGA Pgen + モデル別 PLL（full-seq / CDR3）]
                                         ↓
                          04_benchmark_classification.py → data/benchmarks/classification_results.csv
                          [FLAb2 11 データセット × 5 モデル × 2 プーリング × 2 分類器]
                                         ↓
TheraSAbDab → 05_therapeutic_data.py → data/therapeutic/
                          [sequences_therapeutic.csv, pgen/pll .npy, pgen_pll_scores_therapeutic.csv]
                                         ↓
                          notebook.ipynb → figures/（UMAP プロット + Pgen-PLL 散布図 + 治療用抗体オーバーレイ）
```

### 主要 CSV カラム（`data/sequences_processed.csv`）

| カラム | 内容 |
|--------|------|
| `vh_seq` | 重鎖 VH アミノ酸配列 |
| `vl_seq` | 軽鎖 VL アミノ酸配列 |
| `v_family` | V 遺伝子ファミリー（例: `IGHV1`） |
| `v_call_heavy`, `j_call_heavy` | V/J アレル名（OLGA で使用） |
| `cdr3_aa_heavy` | CDR-H3 配列（OAS 形式: 保存 C と W を除く） |
| `cdr3_indices` | `vh_seq` 内の CDR-H3 の `(start, end)` インデックス |

### 埋め込み設計

全モデルは**全長配列**を入力として受け取る。CDR-H3 埋め込みは出力残基ベクトルのスライスで抽出する（CDR-H3 単独入力ではない）。

- `*_seq_vecs.npy`: 全残基の平均プーリング → (N, D)
- `*_cdr3_vecs.npy`: CDR-H3 残基の平均プーリング → (N, D)、インデックス無効時は全残基平均にフォールバック

### AbLang-2 トークン化（重要）

AbLang-2 は `w_extra_tkns=True` を使う必要がある。トークン配置:
```
ペア入力:     [<(0), VH[0..L-1](1..L_vh), >(L_vh+1), |(L_vh+2), <, VL..., >]
重鎖のみ入力: [<(0), VH[0..L-1](1..L_vh), >(L_vh+1), |]
```
VH 残基は**常にインデックス 1 〜 1+len(VH)** に位置する。重鎖のみ入力時は `vl=""` を渡して同じトークン構造を維持する。

> **過去のバグ**: 旧コードは `w_extra_tkns=False` で `f"{vh}|{vl}"` 形式を使用し、VH の最初の残基が欠落し `|` セパレータが抽出に含まれていた。現在のコードは修正済み。

### OLGA CDR3 フォーマット変換

OAS の `cdr3_aa_heavy` は保存 C（N 末端）と W/F（C 末端）を除外している。OLGA は両方を必要とする:
```python
cdr3_olga = 'C' + cdr3_aa_heavy + vh_seq[cdr3_end]
```

## `notebook.ipynb` の編集

ノートブックツールではなく JSON を直接操作して編集する:

```python
import json
with open('notebook.ipynb') as f:
    nb = json.load(f)
nb['cells'][i]['source'] = "new code"
with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

## パイプラインへの新モデル追加

1. `02_generate_embeddings.py` に `run_<model>()` 関数を追加（`pool_seq` / `pool_cdr3` を再利用）
2. `main()` の `models_to_run` ディスパッチに追加
3. `--model` のヘルプテキストを更新
4. `notebook.ipynb` の `vectors` 辞書と `MODEL_NAMES` リストに追加

## データの再生成

```bash
# 配列数変更または完全再実行
rm data/sequences_processed.csv data/*.npy
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/01_download_data.py --n <N>
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/02_generate_embeddings.py

# Pgen/PLL のみ再実行
rm data/pgen.npy data/pgen_vj.npy data/*_pll.npy data/pgen_pll_scores.csv
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py

# ベンチマーク分類のみ再実行
rm -rf data/benchmarks/
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/04_benchmark_classification.py

# 治療用抗体データのみ再実行
rm -rf data/therapeutic/
KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/05_therapeutic_data.py
```

## 既知の問題

| エラー | 原因 | 対処 |
|--------|------|------|
| `OMP: Error #15` | macOS での OpenMP 多重ロード | `KMP_DUPLICATE_LIB_OK=TRUE` を設定 |
| `AttributeError: 'AntiBERTy'...all_tied_weights_keys` | transformers v5 との非互換 | `pip install "transformers==4.46.3"` |
| OAS API 404 | 旧 `/api/pairednumbers/` エンドポイント廃止 | 現コードは `ngsdb/paired/` を直接クロール |
| AbLang 重みダウンロードがハング | `requests.get()` が大容量ファイルでハング | `curl` で手動ダウンロードして展開 |
| `Pgen = 0` が多数 | CDR3 末端残基が W/F でない、または V/J アレルが OLGA にない | 正常動作 — `NaN` として記録、log10 変換時に自動除外 |
