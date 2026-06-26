import os
import logging
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

logger = logging.getLogger(__name__)

# ── Argument dataclasses ───────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="tencent/Hunyuan-MT-7B",
        metadata={"help": "HuggingFace model id or local path"},
    )
    # Full BF16 LoRA: base model loaded in bf16, no quantization.
    compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "dtype to load the base model in"},
    )


@dataclass
class DataArguments:
    data_dir: str = field(
        default="/gpfs/home2/tvarelanunes/llm-translate/MasterThesis/pipeline/processed/es-fr",
        metadata={"help": "Directory with train.jsonl / valid.jsonl"},
    )
    max_seq_length: int = field(default=1024)
    preprocessing_num_workers: int = field(default=8)


@dataclass
class LoraArguments:
    lora_r:          int   = field(default=16)
    lora_alpha:      int   = field(default=32)
    lora_dropout:    float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of modules to apply LoRA to"},
    )


# ── Tokenisation ───────────────────────────────────────────────────────────────

def tokenize_example(example, tokenizer, max_length):
    """
    Build a training example from a sharegpt-style row:
        example["messages"]   : list[{"role": ..., "content": ...}]
        example["completion"] : str (the target translation)

    Apply Hunyuan-MT's chat template to the prompt
    """
    messages   = example["messages"]
    completion = example["completion"]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    # Some templates return a batched 2D list/tensor; flatten if so.
    if len(prompt_ids) > 0 and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]

    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels    = [-100] * len(prompt_ids) + completion_ids

    truncated = len(input_ids) > max_length
    input_ids = input_ids[:max_length]
    labels    = labels[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "truncated":      truncated,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARNING,
    )
    logger.info(f"Training args: {training_args}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",   # important for causal LM training
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (full BF16, no quantization) ──────────────────────────────────────
    logger.info("Loading base model in BF16 (full precision LoRA, no quantization) …")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        local_rank = training_args.local_rank if training_args.local_rank != -1 else 0

    compute_dtype = getattr(torch, model_args.compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map={"": local_rank},   
        trust_remote_code=True,
        dtype=compute_dtype,
    )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # use_cache must be off when gradient checkpointing is on.
    model.config.use_cache = not training_args.gradient_checkpointing

    # ── LoRA ──────────────────────────────────────────────────────────────────
    target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    if training_args.local_rank in (-1, 0):
        model.print_trainable_parameters()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
    data_files = {
        "train": os.path.join(data_args.data_dir, "train.jsonl"),
        "validation": os.path.join(data_args.data_dir, "valid.jsonl"),
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # Tally truncations across the whole split so silent truncation can't hide.
    trunc_counter = {"train": 0, "validation": 0}

    def make_preprocess(split_name):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "labels": []}
            for messages, completion in zip(examples["messages"], examples["completion"]):
                ex = tokenize_example(
                    {"messages": messages, "completion": completion},
                    tokenizer,
                    data_args.max_seq_length,
                )
                if ex["truncated"]:
                    trunc_counter[split_name] += 1
                for k in results:
                    results[k].append(ex[k])
            return results
        return preprocess

    tokenized = {}
    for split_name, ds in raw_datasets.items():
        tokenized[split_name] = ds.map(
            make_preprocess(split_name),
            batched=True,
            num_proc=1, 
            remove_columns=ds.column_names,
            desc=f"Tokenising {split_name}",
        )
        n = len(tokenized[split_name])
        t = trunc_counter[split_name]
        logger.info(f"[{split_name}] {n} examples, {t} truncated "
                    f"({100*t/max(1,n):.2f}%) at max_seq_length={data_args.max_seq_length}")

    # ── Data collator ─────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training …")
    train_result = trainer.train()

    trainer.save_model()
    trainer.save_state()

    if training_args.local_rank in (-1, 0):
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info("Training complete.")


if __name__ == "__main__":
    main()