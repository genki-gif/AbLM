#!/usr/bin/env bash
# FLAb2 データ取得スクリプト
# リポジトリ: https://github.com/Graylab/FLAb
# 論文: Chungyoun & Gray, bioRxiv 2025 (DOI: 10.64898/2025.12.27.696706)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/repo"

if [ -d "${REPO_DIR}" ]; then
    echo "[FLAb2] repo already exists at ${REPO_DIR}, skipping clone."
    echo "        To re-download, run: rm -rf ${REPO_DIR} && bash $0"
else
    echo "[FLAb2] Cloning FLAb repository..."
    git clone https://github.com/Graylab/FLAb "${REPO_DIR}"
    echo "[FLAb2] Done. Data saved to ${REPO_DIR}"
fi

echo ""
echo "[FLAb2] Directory structure:"
ls "${REPO_DIR}/" 2>/dev/null || true
