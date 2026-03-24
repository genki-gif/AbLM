#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=24:00:00
#$ -N ablm_bench
#$ -o job/job_logs/
#$ -e job/job_logs/
#$ -m abe
#$ -M honda.g.aa@m.titech.ac.jp

# --- 環境設定 ---
. /etc/profile.d/modules.sh
source ~/.bashrc
conda activate ablm

echo "使用ノード: $(hostname)"
echo "開始時刻: $(date)"

export KMP_DUPLICATE_LIB_OK=TRUE
python scripts/04_benchmark_classification.py \
    --datasets all \
    --model all \
    --batch 32

echo "終了時刻: $(date)"
