# Antibody Benchmarks

このディレクトリには、抗体-抗原複合体予測・評価に使われる代表的なベンチマークの
取得スクリプトと概要をまとめています。
データファイル自体は容量が大きいため `.gitignore` に含め、スクリプトのみ管理します。

## ベンチマーク一覧

| ベンチマーク | 論文 | 測定内容 | 規模 | データ形式 |
|------------|------|---------|------|-----------|
| [FLAb2](#1-flab2) | Chungyoun & Gray, bioRxiv 2025 | 治療用抗体の開発可能性7特性 | 400万+ 抗体、32研究 | CSV（特性別サブディレクトリ） |
| [AbAgym](#2-abagym) | Cia et al., mAbs 2025 | 抗体-抗原複合体のDMSスコア | ~33.5万変異、67 DMS実験 | CSV + PDB |
| [MAGMA-seq](#3-magma-seq) | Petersen et al., Nat Commun 2024 | 抗体Fab変異体の定量的結合親和性 (K_D) | 10 Ab × 9–10 標的 | 処理済みCSV |

---

## 1. FLAb2

- **論文**: Chungyoun & Gray, bioRxiv (2025-12-27)
  DOI: https://doi.org/10.64898/2025.12.27.696706
- **GitHub**: https://github.com/Graylab/FLAb

### 測定内容

治療用抗体の開発可能性（developability）7特性:

| 特性 | 説明 |
|------|------|
| thermostability | 熱安定性（Tm など） |
| expression | 発現量 |
| aggregation | 凝集傾向 |
| binding | 抗原結合能 |
| PK | 薬物動態 (pharmacokinetics) |
| polyreactivity | 多反応性 |
| immunogenicity | 免疫原性 |

### データ形式

```
FLAb/
└── data/
    ├── thermostability/
    ├── expression/
    ├── aggregation/
    ├── binding/
    ├── pk/
    ├── polyreactivity/
    └── immunogenicity/
```

各 CSV の主要カラム: `heavy`, `light`, `fitness`, (特性固有のメタデータ)

### 取得

```bash
bash benchmarks/flab2/download.sh
```

---

## 2. AbAgym

- **論文**: Cia et al., mAbs Vol.17 (2025-11-24)
  DOI: https://doi.org/10.1080/19420862.2025.2592421
- **GitHub**: https://github.com/3BioCompBio/Abagym
- **Zenodo (PDB構造)**: https://zenodo.org/records/17328791

### 測定内容

抗体-抗原複合体の単一アミノ酸置換 (SAV) のDMSスコア。
67のDMS実験から得られた ~33.5万変異を収録。

### データ形式

主要カラム:

| カラム | 説明 |
|--------|------|
| `pdb_file` | PDB ファイル名 |
| `wild_type` | 野生型残基 (1文字) |
| `mutant` | 変異残基 (1文字) |
| `chain` | 抗体鎖 (H/L/A など) |
| `DMS_score` | DMS実験スコア |
| `DMS_score_MinMax` | Min-Max正規化スコア |
| `distance_to_interface` | 界面からの距離 (Å) |

### 取得

```bash
bash benchmarks/abagym/download.sh
```

---

## 3. DASM experiments data (旧 MAGMA-seq スロット)

> **注意**: 当初 MAGMA-seq K_D データとして記載していた Zenodo record 17322891 は、
> 実際には **DASM (Deep Amino acid Selection Models) 論文** のデータセットでした。
> 論文: Matsen IV et al., "Separating mutation from selection in antibody language models"
> GitHub: https://github.com/matsengrp/dasm-experiments

- **Zenodo**: https://zenodo.org/records/17322891 (`dasm-experiments-data.tar.gz`, 123 MB)

### 内容

| ディレクトリ / ファイル | 説明 |
|------------------------|------|
| `FLAb/data/binding/` | Koenig2017_g6, Shanehsazzadeh2023 結合DMS |
| `FLAb/data/expression/` | Koenig2017_g6 発現DMS |
| `*.progen.csv` | ProGen2 スコア済みバリアント |
| `v3/*.csv.gz` | BCR 親子対学習データ (Tang/VanWinkle/Jaffe/Rodriguez) |
| `loris/` | Rodriguez RACE-seq IgBlast アノテーション |

### 本来の MAGMA-seq K_D データについて

Petersen et al. 2024 (DOI: 10.1038/s41467-024-48072-z) の K_D データが必要な場合:
- GitHub: https://github.com/WhiteheadGroup/MAGMA-seq
- SRA: PRJNA1043249, PRJNA1043566, PRJNA1119481
- 生 FASTQ から MAGMA-seq パイプラインで処理が必要

### 取得

```bash
bash benchmarks/magma_seq/download.sh
```

---

## 利用上の注意

- データファイル (`repo/`, `data/`) は `.gitignore` に含まれています。
  リポジトリをクローンした後、各 `download.sh` を実行してください。
- FLAb2 と AbAgym は GitHub リポジトリごとクローンします (`git clone`)。
- MAGMA-seq の処理済みデータは Zenodo からダウンロードします。
  SRA の生 FASTQ は不要です（処理済み zip で十分）。
