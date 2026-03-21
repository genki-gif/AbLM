#!/usr/bin/env bash
# AbAgym データ取得スクリプト
# リポジトリ: https://github.com/3BioCompBio/Abagym
# Zenodo (PDB構造): https://zenodo.org/records/17328791
# 論文: Cia et al., mAbs 2025 (DOI: 10.1080/19420862.2025.2592421)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/repo"
DATA_DIR="${SCRIPT_DIR}/data"

# --- 1. GitHub リポジトリのクローン ---
if [ -d "${REPO_DIR}" ]; then
    echo "[AbAgym] repo already exists at ${REPO_DIR}, skipping clone."
    echo "         To re-download, run: rm -rf ${REPO_DIR} && bash $0"
else
    echo "[AbAgym] Cloning Abagym repository..."
    git clone https://github.com/3BioCompBio/Abagym "${REPO_DIR}"
    echo "[AbAgym] GitHub clone done."
fi

# --- 2. GitHub リポジトリ内の ZIP を解凍 ---
# CSV データ・PDB ファイルはリポジトリ内に含まれているため Zenodo は不要
cd "${REPO_DIR}"

if [ ! -f "AbAgym_data_full.csv" ]; then
    echo "[AbAgym] Unzipping AbAgym_data_full.csv.zip..."
    unzip -q AbAgym_data_full.csv.zip
fi

if [ ! -f "AbAgym_data_non-redundant.csv" ]; then
    echo "[AbAgym] Unzipping AbAgym_data_non-redundant.csv.zip..."
    unzip -q AbAgym_data_non-redundant.csv.zip
fi

if [ ! -d "DMS_big_table_PDB_files" ]; then
    echo "[AbAgym] Unzipping PDB_files.zip..."
    unzip -q PDB_files.zip
fi

echo "[AbAgym] CSV and PDB files ready."
echo ""
echo "NOTE: Zenodo record 17328791 contains FoldX structures (37.4 GB)."
echo "      Download manually if needed:"
echo "      wget -c https://zenodo.org/records/17328791/files/foldx.zip"

echo ""
echo "[AbAgym] Done. Files:"
ls "${REPO_DIR}/" 2>/dev/null || true
ls "${DATA_DIR}/" 2>/dev/null || true
