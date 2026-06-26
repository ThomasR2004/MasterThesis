#!/bin/bash
#SBATCH --job-name=hunyuan-mt-prep-multi
#SBATCH --output=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/prep_%j.out
#SBATCH --error=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/prep_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --partition=genoa


CONTAINER="/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif"
PROJECT_DIR="MasterThesis/pipeline/"


PROCESSED_DATA="MasterThesis/pipeline/processed"


PAIRS="all"
SPLITS="all"


echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Project dir: $PROJECT_DIR"
echo "Pairs: $PAIRS   Splits: $SPLITS"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROCESSED_DATA"

cd "$PROJECT_DIR" || exit 1

export PYPACKAGES="/gpfs/home2/$USER/py_packages"
mkdir -p "$PYPACKAGES"


export HF_HOME="/scratch-shared/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"


echo "=== Preprocessing data (pairs=$PAIRS, splits=$SPLITS) ==="
apptainer exec \
    --nv \
    --bind "$PROJECT_DIR":/workspace \
    --bind "/scratch-shared/$USER":/scratch \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env PYTHONPATH="$PYPACKAGES:/scratch/transformers_upgrade:$PYTHONPATH" \
    "$CONTAINER" \
    python3 /workspace/prepare_data.py --pair "$PAIRS" --split "$SPLITS"

echo "Job finished at: $(date)"
