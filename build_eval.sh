#!/bin/bash
#SBATCH --job-name=build_val
#SBATCH --output=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/build_val_%j.out
#SBATCH --error=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/build_val_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --partition=gpu_h100


TRANSLATIONS_DIR="pipeline/translations_base"

OUT_DIR="pipeline/data/test_data/grpo"


METHOD="rasta"
MODEL="Hunyuan-MT-7B"
BASE_MODEL="mistralai/Mistral-7B-v0.1"

# Allowed: all | politeness | intimacy | formal
STYLE_FILTER="all"

export HF_TOKEN=

CONTAINER="vllm_hunyuan1.sif"
PROJECT_DIR="$SLURM_SUBMIT_DIR"
# ---------------------

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Translations : $TRANSLATIONS_DIR"
echo "Out dir      : $OUT_DIR"
echo "Style filter : $STYLE_FILTER"
echo "Project dir  : $PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUT_DIR"

export PYPACKAGES="/gpfs/home2/$USER/py_packages"
mkdir -p "$PYPACKAGES"
cd "$PROJECT_DIR" || exit 1

export HF_HOME="/scratch-shared/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export PYTHONPATH="/scratch/transformers_upgrade:$PYTHONPATH"

apptainer exec \
    --nv \
    --bind "$PROJECT_DIR":/workspace \
    --bind "/scratch-shared/$USER":/scratch \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env HF_TOKEN="$HF_TOKEN" \
    --env PYTHONPATH="$PYPACKAGES:/scratch/transformers_upgrade:$PYTHONPATH" \
    "$CONTAINER" \
    bash -c "
        pip install --target '$PYPACKAGES' --no-deps peft bitsandbytes accelerate safetensors && \
        python3 /workspace/build_train_examples_fixed.py \
            --translations_dir '$TRANSLATIONS_DIR' \
            --out_dir          '$OUT_DIR' \
            --style            '$STYLE_FILTER' \
            --method           '$METHOD' \
            --model            '$MODEL'
    "

echo "Job finished at: $(date)"