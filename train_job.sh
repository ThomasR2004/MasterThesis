#!/bin/bash
#SBATCH --job-name=hunyuan-mt-lora-en-es
#SBATCH --output=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/lora_%j.out
#SBATCH --error=/gpfs/home2/%u/llm-translate/MasterThesis/pipeline/logs/lora_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100

=
MODEL_CHECKPOINT="tencent/Hunyuan-MT-7B"
CONTAINER="/projects/2/managed_datasets/containers/vllm/vllm_25.09.sif"
PROJECT_DIR="$SLURM_SUBMIT_DIR"
PROCESSED_DATA="/MasterThesis/pipeline/processed/es-fr"
OUTPUT_DIR="/MasterThesis/checkpoints/hunyuan-mt-lora-es-fr"


export HF_TOKEN=

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Model: $MODEL_CHECKPOINT"
echo "Project dir: $PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROCESSED_DATA"

cd "$PROJECT_DIR" || exit 1


export PYPACKAGES="$USER/py_packages"
mkdir -p "$PYPACKAGES"

export HF_HOME="/scratch-shared/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

echo "=== Starting LoRA fine-tuning on $(date) ==="
echo "GPUs allocated: $SLURM_GPUS"

apptainer exec \
    --nv \
    --bind "$PROJECT_DIR":/workspace \
    --bind "/scratch-shared/$USER":/scratch \
    --bind "$OUTPUT_DIR":/checkpoints \
    --bind "$PROCESSED_DATA":/data \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
    --env PYTHONPATH="$PYPACKAGES:/scratch/transformers_upgrade:$PYTHONPATH" \
    "$CONTAINER" \
    bash -c "
        pip install --target '$PYPACKAGES' --no-deps --upgrade peft accelerate && \
        torchrun \
            --nproc_per_node=1 \
            --master_port=29503 \
            /workspace/train_lora.py \
            --model_name_or_path '$MODEL_CHECKPOINT' \
            --data_dir /data \
            --max_seq_length 1024 \
            --preprocessing_num_workers 8 \
            --lora_r 16 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
            --output_dir /checkpoints \
            --num_train_epochs 3 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --gradient_checkpointing True \
            --optim adamw_torch \
            --learning_rate 2e-4 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.05 \
            --weight_decay 0.01 \
            --bf16 True \
            --tf32 True \
            --logging_steps 50 \
            --save_strategy no \
            --report_to none \
            --ddp_find_unused_parameters False \
            --dataloader_num_workers 4 \
            --seed 1
    "

echo "=== Training finished on $(date) ==="