#!/usr/bin/env python

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from trl import GRPOConfig, GRPOTrainer

from grpo_reward import StyleTargetReward


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True, help="e.g. tencent/Hunyuan-MT-7B")
    p.add_argument("--adapter_path", required=True, help="SFT LoRA adapter to continue.")
    p.add_argument("--train_file", required=True,
                  help="JSONL with columns: prompt, target (float[0,1]), source_text.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--attn_implementation", default="sdpa",
                  choices=["sdpa", "eager", "flash_attention_2"])

    # reward / style head
    p.add_argument("--style", default="politeness")
    p.add_argument("--target_lang", required=True,
                  help="Target language key for the style head, e.g. 'zh'. One head per run.")
    p.add_argument("--reward_dir", default="/workspace/")
    p.add_argument("--adequacy_threshold", type=float, default=0.5,
                  help="bge-m3 source<->candidate cosine below which a sample is gated "
                       "to --reward_floor. TUNE THIS on a sample first (see notes).")
    p.add_argument("--reward_floor", type=float, default=-2.0,
                  help="Reward for gated (off-topic) samples; must be below the worst "
                       "possible style reward (-1) so off-topic can never win a group.")
    p.add_argument("--label_scale", type=float, default=4.0)

    # GRPO / sampling
    p.add_argument("--num_generations", type=int, default=6,
                  help="Group size G: completions sampled per prompt.")
    p.add_argument("--temperature", type=float, default=1.0,
                  help="Higher than DPO scoring; we WANT in-group spread to learn from.")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_prompt_length", type=int, default=768)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--kl_beta", type=float, default=0.04,
                  help="KL penalty to the reference (the adapter-disabled base, as in DPO).")

    # vLLM generation (recommended on multi-GPU)
    p.add_argument("--use_vllm", action="store_true", default=True)
    p.add_argument("--no_vllm", dest="use_vllm", action="store_false")
    p.add_argument("--vllm_mode", default="colocate", choices=["colocate", "server"])
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--vllm_server_host", default="127.0.0.1")
    p.add_argument("--vllm_server_port", type=int, default=8000)
  
    # LoRA (only if reinit)
    p.add_argument("--reinit_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,
                  default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # optimisation
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-6,
                  help="GRPO LRs are small, similar to DPO (1e-6..5e-6).")
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=500)
    
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]   # drop token_type_ids

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    base.config.use_cache = False

    if args.reinit_lora:
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none", task_type="CAUSAL_LM",
        )
        model = base
    else:
        # Continue the SFT adapter as a TRAINABLE policy (is_trainable=True is essential).
        model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
        peft_config = None

    train_ds = load_dataset("json", data_files={"train": args.train_file})["train"]
    needed = {"prompt", "target", "source_text"}
    missing = needed - set(train_ds.column_names)
    if missing:
        raise SystemExit(f"train_file is missing required columns: {missing}. "
                         f"Build it with build_grpo_prompts.py.")

    # Reward model + BGE + calibration all load here and stay resident.
    reward_model = StyleTargetReward(
        style=args.style,
        target_lang=args.target_lang,
        reward_dir=args.reward_dir,
        adequacy_threshold=args.adequacy_threshold,
        floor=args.reward_floor,
        label_scale=args.label_scale,
    )
    
    def grpo_reward(prompts, completions, **kwargs):
        # conversational datasets hand back [{"role":"assistant","content": "..."}]
        completions = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
        return reward_model(prompts=prompts, completions=completions, **kwargs)
        
        reward_funcs = [grpo_reward]


    cfg_kwargs = dict(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_steps=args.max_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.kl_beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ddp_find_unused_parameters=False,  
        steps_per_generation=6,         
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        tf32=True,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,   
        seed=args.seed,
    )
    accepted = {name for name, f in GRPOConfig.__dataclass_fields__.items() if f.init}
    dropped = sorted(k for k in cfg_kwargs if k not in accepted)
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in accepted}
    if dropped:
        import trl as _trl
        print(f"[grpo] NOTE: TRL {_trl.__version__} GRPOConfig does not accept "
              f"{dropped}; dropping them.")
        for k in dropped:
            if k == "use_vllm" or k.startswith("vllm"):
                print(f"[grpo]   '{k}' dropped -> the vLLM generation path may differ "
                      f"in this TRL (it may need a separate `trl vllm-serve` process "
                      f"or a different field name). Generation could fall back to slow "
                      f"HF sampling -- confirm before a long run.")
            if k in ("num_generations", "beta", "temperature"):
                print(f"[grpo]   '{k}' dropped -> a CORE GRPO knob is unset; find its "
                      f"field name for your TRL version before trusting results.")
    cfg = GRPOConfig(**cfg_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs = [grpo_reward],
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"GRPO training complete. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()