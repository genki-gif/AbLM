#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
#$ -N ablm_pll
#$ -t 1-7
#$ -o job/job_logs/
#$ -e job/job_logs/
#$ -m abe
#$ -M honda.g.aa@m.titech.ac.jp

# --- 環境設定 ---
. /etc/profile.d/modules.sh
source ~/.bashrc
conda activate ablm

# --- アレイジョブのタスク番号をモデル名にマッピング ---
# SGE_TASK_ID: 1=pgen, 2=antiberty, 3=ablang, 4=ablang2, 5=ablang2heavy, 6=esm2, 7=progen2
MODELS=(pgen antiberty ablang ablang2 ablang2heavy esm2 progen2)
MODEL=${MODELS[$((SGE_TASK_ID - 1))]}

echo "[$SGE_TASK_ID] モデル: $MODEL"
echo "使用ノード: $(hostname)"
echo "開始時刻: $(date)"

export KMP_DUPLICATE_LIB_OK=TRUE
python scripts/03_compute_pgen_pll.py \
    --model "$MODEL" \
    --batch 32

echo "終了時刻: $(date)"
