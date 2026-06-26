#!/bin/bash
#SBATCH --job-name=rasta_base_translation
#SBATCH --output=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/rasta_base_%j.out
#SBATCH --error=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/rasta_base_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100


MODEL_CHECKPOINT="tencent/Hunyuan-MT-7B"

STYLES="politeness intimacy formal"


SPLIT="test"

MAX_SAMPLES=""

export HF_TOKEN=

CONTAINER="vllm_hunyuan1.sif"
PROJECT_DIR="$SLURM_SUBMIT_DIR"


echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Base model:   $MODEL_CHECKPOINT (no adapter)"
echo "GPUs:         $SLURM_GPUS"
echo "Split:        $SPLIT"
echo "Max samples:  ${MAX_SAMPLES:-all}"
echo "Styles:       $STYLES"
echo "Project dir:  $PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/translations"
cd "$PROJECT_DIR" || exit 1


export PYPACKAGES="$USER/py_packages_translate"
mkdir -p "$PYPACKAGES"

rm -f "$PYPACKAGES/.rasta_ready"


export HF_HOME="/scratch-shared/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"


for STYLE in $STYLES; do
    echo ""
    echo "=================================================================="
    echo "=== Base model | Style: $STYLE | started $(date) ==="
    echo "=================================================================="

    apptainer exec \
        --nv \
        --bind "$PROJECT_DIR":/workspace \
        --bind "/scratch-shared/$USER":/scratch \
        --env HF_HOME="$HF_HOME" \
        --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
        --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
        --env HF_TOKEN="$HF_TOKEN" \
        --env PYTHONPATH="$PYPACKAGES:/scratch/transformers_upgrade:$PYTHONPATH" \
        "$CONTAINER" \
        bash -c "
            if [ ! -f '$PYPACKAGES/.rasta_ready' ]; then
                echo '--- wiping contaminated persistent package dir for a clean rebuild ---' && \
                rm -rf '$PYPACKAGES' && mkdir -p '$PYPACKAGES' && \
                echo '--- installing transformers + its COMPLETE pure-Python dep tree ---' && \
                pip install --target '$PYPACKAGES' --upgrade --no-deps \
                    'transformers==4.56.2' \
                    'tokenizers==0.22.2' \
                    'safetensors>=0.4.3' \
                    'huggingface-hub>=0.34.0,<1.0' \
                    'regex' 'requests' 'tqdm>=4.27' 'filelock' \
                    'numpy>=1.17' 'packaging>=20.0' 'pyyaml>=5.1' && \
                echo '--- on-disk versions ---' && \
                ls -d '$PYPACKAGES'/transformers-*.dist-info \
                      '$PYPACKAGES'/tokenizers-*.dist-info \
                      '$PYPACKAGES'/huggingface_hub-*.dist-info 2>/dev/null && \
                touch '$PYPACKAGES/.rasta_ready'
            else
                echo '--- package dir already prepared, skipping rebuild ---'
            fi && \
            echo '--- resolved versions inside container ---' && \
            python3 -c 'import transformers, huggingface_hub; print(\"transformers\", transformers.__version__); print(\"hub\", huggingface_hub.__version__)' && \
            python3 /workspace/rasta_translation.py \
                --model  '$MODEL_CHECKPOINT' \
                --style  '$STYLE' \
                --split  '$SPLIT' \
                ${MAX_SAMPLES:+--max-samples '$MAX_SAMPLES'}
        "

    echo "=== Base model | Style: $STYLE | finished $(date) ==="
done

echo ""
echo "All base-model styles done at: $(date)"