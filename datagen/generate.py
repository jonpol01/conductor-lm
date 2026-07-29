"""Stage-0 synthetic envelope generator.

Pure stdlib. Deterministic per seed. Emits chat-format JSONL (train/eval) plus
a raw eval file (envelope + task_class + gold decision) for oracle replay.

Usage:
  python3 datagen/generate.py --train 12000 --eval 800 --out data/
"""

import argparse
import hashlib
import json
import os
import random

from policy import (LOCAL_CLASSES, MID_CLASSES, FRONTIER_CLASSES, decide)

ALL_CLASSES = sorted(LOCAL_CLASSES | MID_CLASSES | FRONTIER_CLASSES)

# --- task text templates (model must infer the class from text alone) ---
T = {
    "summarize": [
        "Summarize the attached {n}k-token incident log and list the recurring error signatures.",
        "Give me a 10-bullet summary of this design document.",
        "Condense these meeting notes into action items with owners.",
        "添付の会議メモを要約して、決定事項と宿題を箇条書きにしてください。",
    ],
    "classify": [
        "Label each of these {m} support tickets as bug, feature-request, or question.",
        "Classify these commit messages by conventional-commit type.",
        "このレビューコメント一覧をポジティブ/ネガティブ/中立に分類して。",
    ],
    "extract": [
        "Extract every API endpoint and its auth requirement from this OpenAPI spec.",
        "Pull all dates, amounts, and counterparties out of these invoices.",
        "このログから全てのスタックトレースと発生時刻を抽出してください。",
    ],
    "translate": [
        "Translate this release-notes draft into Japanese, keeping technical terms in English.",
        "Translate these {m} UI strings from Japanese to English.",
        "この障害報告書を英語に翻訳して。",
    ],
    "rewrite": [
        "Rewrite this paragraph to be half as long without losing the caveats.",
        "Make this error message friendlier for non-technical users.",
        "この文章をより丁寧なビジネス日本語に書き直してください。",
    ],
    "commit_message": [
        "Write a commit message for this staged diff.",
        "Draft a conventional-commit message for the attached changes.",
    ],
    "changelog": [
        "Generate a changelog entry from these {m} merged PR titles.",
        "Update CHANGELOG-style release notes from this commit range.",
    ],
    "mock_data": [
        "Generate {m} rows of realistic mock customer records as JSON.",
        "Create sample sensor readings (CSV) for a week at 1-minute resolution.",
    ],
    "batch_map": [
        "For each of the {m} files listed, produce a one-line description.",
        "Run the same extraction over every log file in this directory listing.",
        "この{m}件のファイルそれぞれに同じ要約処理をかけてください。",
    ],
    "draft_doc": [
        "Draft an internal FAQ page for the new deployment process.",
        "Write a first-draft README section explaining the plugin system.",
        "新機能の社内向け説明ドキュメントのたたき台を書いてください。",
    ],
    "pr_description": [
        "Write the PR description for this branch diff.",
        "Draft a pull-request body summarizing these commits for reviewers.",
    ],
    "boilerplate_code": [
        "Generate the CRUD scaffolding for a 'projects' resource matching our existing pattern.",
        "Write a pytest fixture file mirroring the one used for the users module.",
        "Add a standard health-check endpoint to this FastAPI service.",
    ],
    "mechanical_refactor": [
        "Rename `getUserInfo` to `fetchUserProfile` across the codebase and fix imports.",
        "Convert these {m} callback-style functions to async/await, no behavior change.",
        "この{m}ファイルのvar宣言を全てconst/letに機械的に置き換えてください。",
    ],
    "light_analysis": [
        "Compare these two benchmark result files and note anything that moved >5%.",
        "Skim this survey data and describe the three most obvious trends.",
    ],
    "second_opinion_review": [
        "Give a second-opinion review of this small diff; the main review already passed.",
        "Sanity-check this migration script that a colleague already approved.",
    ],
    "implement_feature": [
        "Implement retry-with-backoff in the payment webhook handler, with tests.",
        "Add cursor-based pagination to the search API and update the client SDK.",
        "音声認識結果の途中確定をUIに反映する機能を実装してください。",
    ],
    "debug_hard": [
        "The worker deadlocks under load roughly once a day; find the root cause.",
        "This memory leak only appears after ~6h of streaming; diagnose it.",
        "本番でのみ再現するレースコンディションの原因を特定してください。",
    ],
    "architecture_design": [
        "Design the event-sourcing migration plan for the ordering service.",
        "Propose the module boundaries for splitting this monolith; justify trade-offs.",
    ],
    "security_review": [
        "Review this auth middleware change for privilege-escalation risks.",
        "Audit the file-upload endpoint for path traversal and content-type bypasses.",
    ],
    "math_reasoning": [
        "Derive the closed-form expected cost of this retry policy and verify numerically.",
        "Prove whether this scheduling invariant holds under concurrent cancellation.",
    ],
    "multi_step_plan": [
        "Plan the zero-downtime Postgres 14→16 upgrade across three environments.",
        "Lay out a step-by-step rollout plan for the new billing pipeline with rollback points.",
    ],
    "user_facing_critical": [
        "Write the customer-facing incident postmortem for yesterday's outage.",
        "Draft the pricing-change announcement email going to all customers.",
    ],
}

DOMAINS = ["devops", "backend", "frontend", "mobile", "data", "ml", "docs", "support", "billing", "infra"]

FRONTIER_IDS = ["claude-opus", "claude-fable", "frontier-a"]
MID_IDS = ["claude-sonnet", "mid-hosted-a", "mid-hosted-b"]
LOCAL_IDS = ["local-2b", "gemma-e2b-local", "qwen-8b-local", "local-8b"]


def make_fleet(rng, task_class, want_privacy, tokens):
    fleet = []
    f_caps = sorted(set(ALL_CLASSES))
    fleet.append({
        "id": rng.choice(FRONTIER_IDS), "tier": "frontier",
        "cost": 1.0, "caps": f_caps, "ctx_max": rng.choice([200000, 1000000]),
    })
    if rng.random() < 0.85:
        m_caps = sorted(MID_CLASSES | set(rng.sample(sorted(LOCAL_CLASSES), k=rng.randint(5, 9))))
        fleet.append({
            "id": rng.choice(MID_IDS), "tier": "mid",
            "cost": round(rng.uniform(0.25, 0.45), 2),
            "caps": m_caps, "ctx_max": rng.choice([65536, 131072, 200000]),
        })
    has_local = rng.random() < 0.8 or want_privacy
    if has_local:
        l_caps = set(rng.sample(sorted(LOCAL_CLASSES), k=rng.randint(4, 9)))
        if rng.random() < 0.4:  # sometimes a local can draft (budget-downroute lever)
            l_caps |= set(rng.sample(sorted(MID_CLASSES), k=rng.randint(1, 3)))
        if task_class in MID_CLASSES and rng.random() < 0.35:
            l_caps.add(task_class)
        ctx = rng.choice([8192, 16384, 32768])
        if want_privacy:  # privacy envelopes guarantee a capable local executor
            l_caps.add(task_class)
            fitting = [c for c in [8192, 16384, 32768] if c >= tokens]
            ctx = rng.choice(fitting) if fitting else 32768
        fleet.append({
            "id": rng.choice(LOCAL_IDS), "tier": "local",
            "cost": 0.0, "caps": sorted(l_caps), "ctx_max": ctx,
        })
    rng.shuffle(fleet)
    return fleet


def make_envelope(rng):
    task_class = rng.choice(ALL_CLASSES)
    tpl = rng.choice(T[task_class])
    task = tpl.format(n=rng.choice([5, 12, 40, 80]), m=rng.choice([8, 20, 50, 200]))

    tokens = int(10 ** rng.uniform(2.0, 5.2))  # 100 .. ~160k
    flags = []
    if task_class in LOCAL_CLASSES and rng.random() < 0.08:
        flags.append("privacy_required")
    want_privacy = "privacy_required" in flags
    if want_privacy:
        tokens = min(tokens, 30000)
    if task_class == "batch_map" or rng.random() < 0.05:
        flags.append("batch")
    if rng.random() < 0.07 and not want_privacy:
        flags.append("ambiguous")

    crit = rng.choices(["low", "normal", "high", "security"], weights=[25, 55, 12, 8])[0]
    if want_privacy:  # keep privacy examples clean of the criticality override
        crit = rng.choice(["low", "normal"])

    fleet = make_fleet(rng, task_class, want_privacy, tokens)

    history = []
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.4:  # relevant entry exercising the history rule
            history.append({
                "task_class": task_class,
                "tier": rng.choice(["local", "mid"]),
                "outcome": rng.choices(["ok", "escalated", "failed"], weights=[4, 3, 3])[0],
            })
        else:
            history.append({
                "task_class": rng.choice(ALL_CLASSES),
                "tier": rng.choice(["local", "mid", "frontier"]),
                "outcome": rng.choices(["ok", "escalated", "failed"], weights=[7, 2, 1])[0],
            })

    envelope = {
        "task": task,
        "context": {
            "tokens_est": tokens,
            "domain": rng.choice(DOMAINS),
            "artifacts": [rng.choice(["file", "diff", "log", "dataset", "none"])],
            "criticality": crit,
            "flags": flags,
        },
        "fleet": fleet,
        "budget": {"frontier_tokens_remaining": round(
            rng.uniform(0.0, 0.19) if (task_class in MID_CLASSES and rng.random() < 0.25)
            else rng.random(), 2)},
        "history": history,
    }
    return envelope, task_class


def check_decision(d, envelope):
    ids = {e["id"] for e in envelope["fleet"]}
    assert d["route"] in ids, d
    for s in d["plan"]:
        assert s["route"] in ids, d
    assert d["escalation"]["to"] is None or d["escalation"]["to"] in ids, d
    assert 0.0 <= d["confidence"] <= 1.0


def gen_split(n, seed, sys_prompt, seen):
    rng = random.Random(seed)
    chat_rows, raw_rows, rationale_counts = [], [], {}
    while len(raw_rows) < n:
        envelope, task_class = make_envelope(rng)
        key = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        decision = decide(envelope, task_class)
        check_decision(decision, envelope)
        rationale_counts[decision["rationale_class"]] = rationale_counts.get(decision["rationale_class"], 0) + 1
        chat_rows.append({"messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(envelope, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)},
        ]})
        raw_rows.append({"envelope": envelope, "task_class": task_class, "gold": decision})
    return chat_rows, raw_rows, rationale_counts


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--eval", type=int, default=800)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "spec", "system_prompt.txt"), encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    os.makedirs(args.out, exist_ok=True)
    seen = set()
    train_chat, _, tr_counts = gen_split(args.train, args.seed, sys_prompt, seen)
    eval_chat, eval_raw, ev_counts = gen_split(args.eval, args.seed + 1000, sys_prompt, seen)

    write_jsonl(os.path.join(args.out, "train.jsonl"), train_chat)
    write_jsonl(os.path.join(args.out, "eval.jsonl"), eval_chat)
    write_jsonl(os.path.join(args.out, "eval_raw.jsonl"), eval_raw)

    print(f"train={len(train_chat)} eval={len(eval_chat)}")
    print("train rationale distribution:")
    for k, v in sorted(tr_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34s} {v:6d}  ({100*v/len(train_chat):.1f}%)")


if __name__ == "__main__":
    main()
