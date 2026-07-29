"""Stage-0 eval: schema validity + agreement with the gold rule policy.

python eval/eval_v0.py --adapter runs/e2b-v0/final --data data/eval_raw.jsonl --n 200
"""

import argparse
import json
import os

import torch
from jsonschema import Draft7Validator
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))


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
    args = ap.parse_args()

    with open(os.path.join(HERE, "..", "spec", "decision.schema.json"), encoding="utf-8") as f:
        validator = Draft7Validator(json.load(f))
    with open(os.path.join(HERE, "..", "spec", "system_prompt.txt"), encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0},
        attn_implementation="eager", torch_dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]

    n = len(rows)
    parsed = valid = route_ok = rat_ok = fleet_ok = 0
    rat_confusion = {}
    for i, row in enumerate(rows):
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(row["envelope"], ensure_ascii=False)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=320, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        d = extract_json(text)
        gold = row["gold"]
        if d is None:
            rat_confusion[(gold["rationale_class"], "UNPARSEABLE")] = \
                rat_confusion.get((gold["rationale_class"], "UNPARSEABLE"), 0) + 1
            continue
        parsed += 1
        if not list(validator.iter_errors(d)):
            valid += 1
        fleet_ids = {e["id"] for e in row["envelope"]["fleet"]}
        if d.get("route") in fleet_ids:
            fleet_ok += 1
        if d.get("route") == gold["route"]:
            route_ok += 1
        pred_rat = d.get("rationale_class", "MISSING")
        if pred_rat == gold["rationale_class"]:
            rat_ok += 1
        else:
            rat_confusion[(gold["rationale_class"], pred_rat)] = \
                rat_confusion.get((gold["rationale_class"], pred_rat), 0) + 1
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{n}] parsed={parsed} valid={valid} route_acc={route_ok/(i+1):.3f}")

    results = {
        "n": n,
        "parse_rate": parsed / n,
        "schema_valid_rate": valid / n,
        "route_in_fleet_rate": fleet_ok / n,
        "route_accuracy": route_ok / n,
        "rationale_accuracy": rat_ok / n,
    }
    print(json.dumps(results, indent=2))
    if rat_confusion:
        print("top confusions (gold -> pred):")
        for (g, p), c in sorted(rat_confusion.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {g:34s} -> {p:34s} {c}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results,
                       "confusions": [[g, p, c] for (g, p), c in rat_confusion.items()]}, f, indent=2)


if __name__ == "__main__":
    main()
