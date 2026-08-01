"""Interactive smoke-test for a trained Conductor adapter.

Build an envelope from CLI flags (or pass one with --envelope) and print the
model's routing decision, schema-validated.

  python serve/try_decision.py --adapter models/conductor-e2b-v0 \
      --task "Summarise this 40k-token incident log" --tokens 41000

  python serve/try_decision.py --adapter models/conductor-e2b-v0 --demo
"""

import argparse
import json
import os
import sys

import torch
from jsonschema import Draft7Validator
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from common import load_base  # noqa: E402

DEFAULT_FLEET = [
    {"id": "claude-opus", "tier": "frontier", "cost": 1.0, "ctx_max": 200000,
     "caps": ["summarize", "classify", "extract", "translate", "rewrite", "commit_message",
              "changelog", "mock_data", "batch_map", "draft_doc", "pr_description",
              "boilerplate_code", "mechanical_refactor", "light_analysis",
              "second_opinion_review", "implement_feature", "debug_hard",
              "architecture_design", "security_review", "math_reasoning",
              "multi_step_plan", "user_facing_critical"]},
    {"id": "claude-sonnet", "tier": "mid", "cost": 0.35, "ctx_max": 200000,
     "caps": ["draft_doc", "pr_description", "boilerplate_code", "mechanical_refactor",
              "light_analysis", "second_opinion_review", "summarize", "extract", "rewrite"]},
    {"id": "local-2b", "tier": "local", "cost": 0.0, "ctx_max": 32768,
     "caps": ["summarize", "classify", "extract", "translate", "rewrite",
              "commit_message", "changelog", "mock_data", "batch_map"]},
]

DEMOS = [
    ("light local work", dict(
        task="Translate these 20 UI strings from Japanese to English.",
        tokens=800, crit="normal", flags=[], budget=0.6, history=[])),
    ("security-critical", dict(
        task="Review this auth middleware change for privilege-escalation risks.",
        tokens=6000, crit="security", flags=[], budget=0.6, history=[])),
    ("privacy-pinned", dict(
        task="Summarise these internal HR complaint records.",
        tokens=9000, crit="normal", flags=["privacy_required"], budget=0.6, history=[])),
    ("history says mid already failed (the known weak spot)", dict(
        task="Give a second-opinion review of this small diff; the main review passed.",
        tokens=40000, crit="normal", flags=[], budget=0.6,
        history=[{"task_class": "second_opinion_review", "tier": "mid",
                  "outcome": "escalated"}])),
    ("budget nearly exhausted", dict(
        task="Draft an internal FAQ page for the new deployment process.",
        tokens=3000, crit="normal", flags=[], budget=0.05, history=[])),
]


def envelope(task, tokens, crit, flags, budget, history, fleet=None):
    return {
        "task": task,
        "context": {"tokens_est": tokens, "domain": "eng", "artifacts": ["none"],
                    "criticality": crit, "flags": flags},
        "fleet": fleet or DEFAULT_FLEET,
        "budget": {"frontier_tokens_remaining": budget},
        "history": history,
    }


def decide(model, tok, sys_prompt, env, validator):
    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(env, ensure_ascii=False)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=320, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        d = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:
        return None, text, ["unparseable"]
    return d, text, [e.message for e in validator.iter_errors(d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--adapter", default=os.path.join(ROOT, "models", "conductor-e2b-v0"))
    ap.add_argument("--task")
    ap.add_argument("--tokens", type=int, default=5000)
    ap.add_argument("--crit", default="normal",
                    choices=["low", "normal", "high", "security"])
    ap.add_argument("--flags", default="")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--envelope", help="path to a full envelope JSON")
    ap.add_argument("--demo", action="store_true", help="run the built-in cases")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "spec", "decision.schema.json"), encoding="utf-8") as f:
        validator = Draft7Validator(json.load(f))
    with open(os.path.join(ROOT, "spec", "system_prompt.txt"), encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    model, tok = load_base(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    if args.demo:
        cases = DEMOS
    elif args.envelope:
        cases = [("from file", json.load(open(args.envelope, encoding="utf-8")))]
    else:
        if not args.task:
            ap.error("need --task, --envelope or --demo")
        cases = [("cli", dict(task=args.task, tokens=args.tokens, crit=args.crit,
                              flags=[f for f in args.flags.split(",") if f],
                              budget=args.budget, history=[]))]

    tiers = {e["id"]: e["tier"] for e in DEFAULT_FLEET}
    for label, spec in cases:
        env = spec if "fleet" in spec else envelope(**spec)
        d, raw, errs = decide(model, tok, sys_prompt, env, validator)
        print(f"\n\033[1m{label}\033[0m")
        print(f"  task    : {env['task'][:72]}")
        print(f"  context : {env['context']['tokens_est']} tok · "
              f"{env['context']['criticality']} · flags={env['context']['flags'] or '—'} · "
              f"budget={env['budget']['frontier_tokens_remaining']}")
        if env["history"]:
            print(f"  history : {env['history']}")
        if d is None:
            print(f"  \033[31mUNPARSEABLE\033[0m: {raw[:150]}")
            continue
        print(f"  -> route      : {d.get('route')}  [{tiers.get(d.get('route'), '?')}]")
        print(f"     rationale  : {d.get('rationale_class')}")
        print(f"     escalate   : {d.get('escalation', {}).get('to')} "
              f"on {d.get('escalation', {}).get('on')}")
        print(f"     confidence : {d.get('confidence')}")
        print(f"     schema     : {'valid' if not errs else 'INVALID ' + str(errs[:2])}")


if __name__ == "__main__":
    main()
