"""Throughput bench: what does a training step actually cost, and at what batch size?

Stage-0 ran at bs=1 / grad_accum=16 and used only ~6.2 GB of a 10 GB card. This
measures samples/s and peak VRAM across batch sizes so the next run is sized from
data instead of habit.

  python train/bench.py --steps 8 --batch-sizes 1,2,4,8
"""

import argparse
import os
import sys
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load_base  # noqa: E402


def lora_targets(model):
    from bitsandbytes.nn import Linear4bit
    want = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def hit(n, m):
        if not isinstance(m, Linear4bit):
            return False
        p = n.split(".")
        return p[-1] in want or (p[-1] == "linear" and len(p) >= 2 and p[-2] in want)
    return sorted(n for n, m in model.named_modules() if hit(n, m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/johnpaul/models/gemma-4-E2B-it")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seq", type=int, default=704)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--no-liger", action="store_true", help="measure the unfused path")
    args = ap.parse_args()

    model, tok = load_base(args.model)
    model.config.use_cache = False
    # MUST match training: TRL applies Liger, whose fused linear cross-entropy never
    # materialises the [batch, seq, 262144] logits tensor. Without it this bench
    # measures a different memory profile than the job it is supposed to model —
    # the first version peaked at 14.4 GB on a 10 GB card and thrashed.
    if not args.no_liger:
        from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model)
        print("liger applied")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=lora_targets(model)))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    print(f"\n{'bs':>4}{'s/step':>10}{'samples/s':>12}{'peak VRAM':>12}{'vs bs=1':>10}")
    base = None
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(10, 200000, (bs, args.seq), device=model.device)
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
        try:
            for _ in range(2):                       # warmup
                model(**batch).loss.backward()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.steps):
                model(**batch).loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.steps
        except torch.OutOfMemoryError:
            print(f"{bs:>4}{'OOM':>10}")
            break
        sps = bs / dt
        base = base or sps
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{bs:>4}{dt:>10.2f}{sps:>12.2f}{peak:>10.2f} GB{sps/base:>9.2f}x")


if __name__ == "__main__":
    main()
