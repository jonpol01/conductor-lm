"""Stage-0 eval: schema validity + agreement with the gold rule policy.

python eval/eval_v0.py --adapter runs/e2b-v0/final --data data/eval_raw.jsonl --n 200
"""

import argparse
import json
import os
import sys

import torch
from jsonschema import Draft7Validator
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import load_base  # noqa: E402

TIER_RANK = {"local": 0, "mid": 1, "frontier": 2}


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/eval_raw.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=16)
    # longest real decision is 103 tokens; 160 is headroom without a runaway cap
    ap.add_argument("--max-new", type=int, default=160, dest="max_new")
    args = ap.parse_args()

    with open(os.path.join(HERE, "..", "spec", "decision.schema.json"), encoding="utf-8") as f:
        validator = Draft7Validator(json.load(f))
    with open(os.path.join(HERE, "..", "spec", "system_prompt.txt"), encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    model, tok = load_base(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]

    # Batched greedy decoding. Prompts cluster tightly (p50 508 / max 588 tokens) and
    # decisions are short (p50 67 / max 103), so padding waste is small and batching
    # is close to a free multiple. Decoder-only models need LEFT padding or the
    # generated continuation starts after the pad run.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def generate_batch(batch_rows):
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": json.dumps(r["envelope"], ensure_ascii=False)}],
                add_generation_prompt=True, tokenize=False)
            for r in batch_rows
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        cut = enc["input_ids"].shape[1]
        return [tok.decode(o[cut:], skip_special_tokens=True) for o in out]

    n = len(rows)
    parsed = valid = route_ok = rat_ok = fleet_ok = 0
    # fail-up accounting: a misroute toward a MORE capable tier costs money, one
    # toward a cheaper tier risks silently worse output. Only the latter is a
    # safety failure, so they are counted separately rather than lumped as error.
    up = down = lateral = 0
    esc_ok = esc_total = 0
    rat_confusion = {}
    samples = []
    texts = []
    for s in range(0, n, args.batch):
        texts.extend(generate_batch(rows[s:s + args.batch]))
        print(f"[gen {min(s + args.batch, n)}/{n}]", flush=True)

    for i, row in enumerate(rows):
        d = extract_json(texts[i])
        gold = row["gold"]
        if d is None:
            rat_confusion[(gold["rationale_class"], "UNPARSEABLE")] = \
                rat_confusion.get((gold["rationale_class"], "UNPARSEABLE"), 0) + 1
            continue
        parsed += 1
        if not list(validator.iter_errors(d)):
            valid += 1
        fleet = row["envelope"]["fleet"]
        fleet_ids = {e["id"] for e in fleet}
        tier_of = {e["id"]: e["tier"] for e in fleet}
        if d.get("route") in fleet_ids:
            fleet_ok += 1
        if d.get("route") == gold["route"]:
            route_ok += 1
        elif d.get("route") in tier_of:
            pr, gr = TIER_RANK[tier_of[d["route"]]], TIER_RANK[tier_of[gold["route"]]]
            if pr > gr:
                up += 1
            elif pr < gr:
                down += 1
            else:
                lateral += 1
        # escalation must point strictly upward (null only at the frontier)
        esc = d.get("escalation") or {}
        if d.get("route") in tier_of:
            esc_total += 1
            to = esc.get("to")
            here = TIER_RANK[tier_of[d["route"]]]
            if to is None:
                esc_ok += 1 if here == 2 else 0
            elif to in tier_of and TIER_RANK[tier_of[to]] > here:
                esc_ok += 1
        pred_rat = d.get("rationale_class", "MISSING")
        if pred_rat == gold["rationale_class"]:
            rat_ok += 1
        else:
            rat_confusion[(gold["rationale_class"], pred_rat)] = \
                rat_confusion.get((gold["rationale_class"], pred_rat), 0) + 1
        samples.append({"i": i, "task_class": row["task_class"],
                        "gold_route": gold["route"], "pred_route": d.get("route"),
                        "gold_rationale": gold["rationale_class"], "pred_rationale": pred_rat,
                        # P2.2: is confidence low precisely where the model is wrong?
                        "confidence": d.get("confidence")})
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{n}] parsed={parsed} valid={valid} route_acc={route_ok/(i+1):.3f}")

    results = {
        "n": n,
        "parse_rate": parsed / n,
        "schema_valid_rate": valid / n,
        "route_in_fleet_rate": fleet_ok / n,
        "route_accuracy": route_ok / n,
        "rationale_accuracy": rat_ok / n,
        "misroute_up": up,
        "misroute_down": down,
        "misroute_lateral": lateral,
        "unsafe_downroute_rate": down / n,
        "escalation_direction_valid": (esc_ok / esc_total) if esc_total else None,
    }
    print(json.dumps(results, indent=2))
    if rat_confusion:
        print("top confusions (gold -> pred):")
        for (g, p), c in sorted(rat_confusion.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {g:34s} -> {p:34s} {c}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results,
                       "confusions": [[g, p, c] for (g, p), c in rat_confusion.items()],
                       "samples": samples}, f, indent=2)


if __name__ == "__main__":
    main()
