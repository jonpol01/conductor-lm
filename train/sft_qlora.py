"""Stage-0 QLoRA SFT for Conductor on a single 10GB GPU (RTX 3080-class).

python train/sft_qlora.py --data data --out runs/e2b-v0
"""

import argparse
import os
import sys

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load_base  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/e2b-v0")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-len", type=int, default=1536)
    # bs=4 measured 6.7x the samples/s of bs=1 on a 10GB card (train/bench.py);
    # throughput peaks at 4 and falls off above it as VRAM spills to system memory.
    # grad_accum keeps the effective batch at 16, unchanged from the Stage-0 run.
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1, dest="max_steps",
                    help="cap optimizer steps — use 5 for the pre-flight smoke run")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    model, tok = load_base(args.model, attn=args.attn)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # gemma4 wraps some projections in Gemma4ClippableLinear (real Linear4bit at
    # <proj>.linear) and has plain Linear4bit in other layers — select targets by
    # module type so LoRA lands only on actual linears, text stack only.
    from bitsandbytes.nn import Linear4bit
    WANT = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def is_target(name, mod):
        if not isinstance(mod, Linear4bit):
            return False
        if "vision_tower" in name or "audio_tower" in name:
            return False
        parts = name.split(".")
        if parts[-1] in WANT:
            return True
        return parts[-1] == "linear" and len(parts) >= 2 and parts[-2] in WANT

    targets = sorted(n for n, m in model.named_modules() if is_target(n, m))
    if not targets:
        raise RuntimeError("no LoRA target modules found")
    print(f"LoRA targets: {len(targets)} modules")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={
        "train": f"{args.data}/train.jsonl",
        "eval": f"{args.data}/eval.jsonl",
    })

    cfg_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bs,
        # Was pinned to 1 after the default of 8 spiked VRAM past 10GB. With bs=4
        # proven to fit, match it: the probe spent 852s of a 1027s run inside eval.
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        optim="adamw_8bit",
        use_liger_kernel=True,  # fused CE: avoids materializing seq×262k-vocab logits
        report_to="none",
    )
    try:
        cfg = SFTConfig(assistant_only_loss=True, **cfg_kwargs)
        trainer = SFTTrainer(model=model, args=cfg, processing_class=tok,
                             train_dataset=ds["train"], eval_dataset=ds["eval"])
    except Exception as e:  # chat template without generation markers, older trl, etc.
        print(f"assistant_only_loss unavailable ({e}); training on full sequence")
        cfg = SFTConfig(**cfg_kwargs)
        trainer = SFTTrainer(model=model, args=cfg, processing_class=tok,
                             train_dataset=ds["train"], eval_dataset=ds["eval"])

    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print("saved:", args.out + "/final")


if __name__ == "__main__":
    main()
