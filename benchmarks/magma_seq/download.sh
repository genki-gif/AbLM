#!/usr/bin/env bash
# MAGMA-seq データ取得スクリプト (処理済みデータ)
# Zenodo: https://zenodo.org/records/17322891
#   ※ Petersen 2024 + Kirby 2025 のデータをML評価用に再処理済み
# パイプライン GitHub: https://github.com/WhiteheadGroup/MAGMA-seq
# 論文: Petersen et al., Nat Commun 15, 3974 (2024) (DOI: 10.1038/s41467-024-48072-z)
#
# 注意: SRA の生 FASTQ (PRJNA1043249, PRJNA1043566, PRJNA1119481) は不要。
#       処理済み CSV の zip をダウンロードするだけで十分です。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

mkdir -p "${DATA_DIR}"

ZENODO_URL="https://zenodo.org/records/17322891/files/dasm-experiments-data.tar.gz"
ARCHIVE_FILE="${DATA_DIR}/dasm-experiments-data.tar.gz"

# --- Zenodo からダウンロード ---
if [ -f "${ARCHIVE_FILE}" ]; then
    echo "[MAGMA-seq] ${ARCHIVE_FILE} already exists, skipping download."
    echo "            To re-download, run: rm ${ARCHIVE_FILE} && bash $0"
else
    echo "[MAGMA-seq] Downloading processed data from Zenodo..."
    wget -c -O "${ARCHIVE_FILE}" "${ZENODO_URL}"
    echo "[MAGMA-seq] Download done: ${ARCHIVE_FILE}"
fi

# --- 解凍 ---
if [ ! -d "${DATA_DIR}/dasm-experiments-data" ]; then
    echo "[MAGMA-seq] Extracting ${ARCHIVE_FILE}..."
    tar -xzf "${ARCHIVE_FILE}" -C "${DATA_DIR}/"
    echo "[MAGMA-seq] Extraction done."
else
    echo "[MAGMA-seq] Already extracted at ${DATA_DIR}/dasm-experiments-data, skipping."
fi

echo ""
echo "[MAGMA-seq] Done. Files:"
ls "${DATA_DIR}/" 2>/dev/null || true
ls "${DATA_DIR}/dasm-experiments-data/" 2>/dev/null || true
