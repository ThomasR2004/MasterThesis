#!/bin/bash
#SBATCH --job-name=rasta_grpo
#SBATCH --output=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/grpo_%j.out
#SBATCH --error=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/grpo_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100

MODEL_CHECKPOINT="tencent/Hunyuan-MT-7B"


SFT_ADAPTER="/MasterThesis/checkpoints/hunyuan-mt-lora-it-pt"

# Direction + style for this run. One target-language style head per run.
STYLE="formal"
SRC_LANG="pt"
TGT_LANG="it"


REWARD_DIR="/workspace/"


ADEQUACY_THRESHOLD="0.5"
REWARD_FLOOR="-2.0"


NUM_GENERATIONS="6"      
TEMPERATURE="1.0"        
KL_BETA="0.04"
LR="1e-6"
EPOCHS="1.0"
PER_DEVICE_BS="8"
GRAD_ACCUM="4"
USE_VLLM="server"
VLLM_GPU_MEM_UTIL="0.3"   


SCHEDULE="grid"          
TARGET_LEVELS="0,1"
LABEL_SCALE="1.0"


MAX_SAMPLES=""
VLLM_PORT="8000"
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1


export HF_TOKEN=


CONTAINER="vllm_hunyuan1.sif"
PROJECT_DIR="$SLURM_SUBMIT_DIR"


echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Base model:   $MODEL_CHECKPOINT"
echo "SFT adapter:  $SFT_ADAPTER"
echo "Direction:    ${SRC_LANG}->${TGT_LANG}  | style=$STYLE"
echo "GPUs:         $SLURM_GPUS"
echo "Max samples:  ${MAX_SAMPLES:-all}"
echo "Project dir:  $PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/grpo_data" "$PROJECT_DIR/checkpoints"
cd "$PROJECT_DIR" || exit 1


export PYPACKAGES="/gpfs/home2/$USER/py_packages_grpo"
mkdir -p "$PYPACKAGES"

export HF_HOME="/scratch-shared/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

PROMPT_FILE="$PROJECT_DIR/grpo_data/${SRC_LANG}_${TGT_LANG}_${STYLE}_${SCHEDULE}.jsonl"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/grpo_${SRC_LANG}_${TGT_LANG}_${STYLE}"


SFT_BIND=""
if [ -n "$SFT_ADAPTER" ]; then
    SFT_BIND="--bind $SFT_ADAPTER:$SFT_ADAPTER"
fi

apptainer exec --nv \
    --bind "$PROJECT_DIR":/workspace \
    --bind "/scratch-shared/$USER":/scratch \
    --env HF_HOME="$HF_HOME" --env HF_TOKEN="$HF_TOKEN" \
    --env PYTHONPATH="$PYPACKAGES:/workspace" \
    --env PATH="$PYPACKAGES/bin:$PATH" \
    "$CONTAINER" bash -c "
        set -e
        VERIFY='import multiprocess, dill, xxhash, pyarrow, fsspec, aiohttp, pandas, six, rich; import datasets, trl, peft, accelerate, bitsandbytes, transformers; print(\"OK: trl\", trl.__version__, \"| datasets\", datasets.__version__, \"| transformers\", transformers.__version__)'
        if python3 -c \"\$VERIFY\" >/dev/null 2>&1; then
            echo '--- deps already importable, skipping install ---'
        else
            echo '--- (re)building package dir with the full --no-deps closure ---'
            # Wipe any partial/broken state (the old touch-file readiness was a lie:
            # it meant 'install ran', not 'imports work'). --ignore-installed forces
            # every NAMED package into the target even when the container claims to
            # satisfy it (that satisfied-skip is why multiprocess never landed).
            # --no-deps + explicit closure keeps pip from ever pulling torch/vllm,
            # which must come from the container.
            rm -rf '$PYPACKAGES' && mkdir -p '$PYPACKAGES'
            pip install --target '$PYPACKAGES' --upgrade --no-deps --ignore-installed \
                'transformers==4.56.2' 'tokenizers==0.22.2' 'safetensors>=0.4.3' \
                'huggingface-hub>=0.34.0,<1.0' \
                'regex' 'requests' 'tqdm>=4.27' 'filelock' 'numpy>=1.17' \
                'packaging>=20.0' 'pyyaml>=5.1' 'typing-extensions>=4.10' 'psutil' \
                'trl==1.5.1' 'peft>=0.11' 'accelerate>=0.30' \
                'datasets>=2.19' 'bitsandbytes>=0.43' \
                'pyarrow>=15' 'dill' 'multiprocess' 'xxhash' 'fsspec>=2023.1.0' 'pandas' \
                'aiohttp' 'aiosignal' 'frozenlist' 'multidict' 'yarl' 'attrs' \
                'propcache' 'aiohappyeyeballs' 'async-timeout' \
                'python-dateutil' 'pytz' 'tzdata' 'six' \
                'rich' 'markdown-it-py' 'mdurl' 'pygments'
        fi
        echo '--- verifying the import closure BEFORE the long stages ---'
        python3 -c \"\$VERIFY\"

        # ---------- Stage 1: build the GRPO prompt dataset (CPU-ish, no vLLM) ----------
        if [ ! -f '$PROMPT_FILE' ]; then
            echo '=== Stage 1: building GRPO prompts ==='
            python3 /workspace/build_grpo_prompts.py \
                --style '$STYLE' --src_lang '$SRC_LANG' --tgt_lang '$TGT_LANG' \
                --split train --schedule '$SCHEDULE' \
                --target_levels '$TARGET_LEVELS' --label_scale '$LABEL_SCALE' \
                ${MAX_SAMPLES:+--max_samples '$MAX_SAMPLES'} \
                --output '$PROMPT_FILE'
        else
            echo '=== Stage 1: prompt file already exists, skipping ==='
        fi

        echo '=== Stage 2: GRPO training (colocate vLLM, 2 GPUs) ==='
        accelerate launch \
            --num_processes 1 \
            --num_machines 1 \
            --mixed_precision bf16 \
            --dynamo_backend no \
            /workspace/train_grpo.py \
            --model_name_or_path '$MODEL_CHECKPOINT' \
            --adapter_path '$SFT_ADAPTER' \
            --train_file '$PROMPT_FILE' \
            --output_dir '$OUTPUT_DIR' \
            --style '$STYLE' \
            --target_lang '$TGT_LANG' \
            --reward_dir '$REWARD_DIR' \
            --adequacy_threshold '$ADEQUACY_THRESHOLD' \
            --reward_floor '$REWARD_FLOOR' \
            --label_scale '$LABEL_SCALE' \
            --num_generations '$NUM_GENERATIONS' \
            --temperature '$TEMPERATURE' \
            --kl_beta '$KL_BETA' \
            --learning_rate '$LR' \
            --num_train_epochs '$EPOCHS' \
            --per_device_train_batch_size '$PER_DEVICE_BS' \
            --gradient_accumulation_steps '$GRAD_ACCUM' \
            --gradient_checkpointing \
            --use_vllm --vllm_mode colocate \
            --vllm_gpu_memory_utilization '$VLLM_GPU_MEM_UTIL'
        "

echo ""
echo "GRPO done at: $(date)"
echo "Adapter saved under: $OUTPUT_DIR"
