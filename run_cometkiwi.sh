#!/bin/bash
#SBATCH --job-name=cometkiwi
#SBATCH --output=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/cometkiwi_%j.out
#SBATCH --error=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/cometkiwi_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_h100


TRANSLATIONS_DIR="/pipeline/translations"
OUT_DIR="/data/test_data/cometkiwi"


COMET_MODEL="Unbabel/wmt22-cometkiwi-da"

# Optional: restrict to one style. all | politeness | intimacy | formal
STYLE_FILTER="all"


export HF_TOKEN=

CONTAINER="vllm_hunyuan1.sif"
PROJECT_DIR="$SLURM_SUBMIT_DIR"


COMET_VENV="/scratch-shared/$USER/comet_venv"
# ---------------------

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Translations : $TRANSLATIONS_DIR"
echo "Out dir      : $OUT_DIR"
echo "Comet model  : $COMET_MODEL"
echo "Style filter : $STYLE_FILTER"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR" || exit 1

export HF_HOME="/scratch-shared/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

apptainer exec \
    --nv \
    --bind "$PROJECT_DIR":/workspace \
    --bind "/scratch-shared/$USER":/scratch \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env HF_TOKEN="$HF_TOKEN" \
    "$CONTAINER" \
    bash -c "
        if [ ! -f '$COMET_VENV/.ready' ]; then
            echo '--- building comet venv (reusing container torch) ---' && \
            python3 -m venv --system-site-packages '$COMET_VENV' && \
            '$COMET_VENV/bin/pip' install --upgrade pip && \
            '$COMET_VENV/bin/pip' install 'unbabel-comet' && \
            touch '$COMET_VENV/.ready'
        else
            echo '--- comet venv already prepared, skipping build ---'
        fi && \
        echo '--- versions ---' && \
        '$COMET_VENV/bin/python' -c 'import torch, comet; print(\"torch\", torch.__version__); print(\"comet OK\")' && \
        '$COMET_VENV/bin/python' /workspace/run_cometkiwi.py \
            --translations_dir '$TRANSLATIONS_DIR' \
            --out_dir          '$OUT_DIR' \
            --comet_model      '$COMET_MODEL' \
            --style            '$STYLE_FILTER'
    "

echo "Job finished at: $(date)"
