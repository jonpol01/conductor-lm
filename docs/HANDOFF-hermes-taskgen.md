# Hermes task: generate realistic task descriptions for Conductor

**One job.** Write ~2,200 short, realistic engineering task descriptions — 100 for each of the
22 task classes below — as JSONL. Nothing else. No training, no routing decisions, no repo edits.

## Why (context, not instructions)

Conductor is a small model that reads a task and decides which tier of LLM should handle it.
Its current training data uses ~4 hand-written templates per class, so the model may be
learning template shapes rather than what the tasks actually mean. Your output replaces those
templates with real variety. **You are writing the task text only — never the routing answer.**

## Endpoint

Use the Grok profile you already have:

```
http://192.168.2.55:8651/v1/chat/completions      (OpenAI-compatible, your existing auth)
```

Fallback if that is unavailable — local, no auth:

```
http://192.168.2.55:1234/v1/chat/completions      model: qwen3.5-9b-mlx
```

## Output

Append one JSON object per line to:

```
/Users/soliva/conductor-taskgen/tasks.jsonl
```

(Use the absolute path. `~` resolves to `/Users/hermes` for you and will land in the wrong place.)

Each line, exactly these two keys:

```json
{"task_class": "summarize", "task": "Condense this 40k-token incident log into the recurring error signatures."}
```

## The 22 classes, and what each means

| class | one-line meaning |
|---|---|
| `summarize` | condense a document, log, or thread |
| `classify` | label items into fixed categories |
| `extract` | pull structured fields out of unstructured text |
| `translate` | between human languages |
| `rewrite` | reword, shorten, change tone |
| `commit_message` | write a commit message from a diff |
| `changelog` | release notes from commits or PRs |
| `mock_data` | generate sample/fake records |
| `batch_map` | apply the same small operation over many items |
| `draft_doc` | first draft of internal documentation |
| `pr_description` | describe a branch's changes for reviewers |
| `boilerplate_code` | scaffolding that follows an existing pattern |
| `mechanical_refactor` | rename/convert with no behaviour change |
| `light_analysis` | skim data and note the obvious trends |
| `second_opinion_review` | sanity-check something already reviewed |
| `implement_feature` | build a real feature, with tests |
| `debug_hard` | diagnose an intermittent or subtle bug |
| `architecture_design` | design decisions with long-lived consequences |
| `security_review` | audit code for vulnerabilities |
| `math_reasoning` | derive or prove something, verify numerically |
| `multi_step_plan` | sequence a multi-stage rollout or migration |
| `user_facing_critical` | text customers will read (postmortem, announcement) |

## Rules for the task text

1. **One sentence, imperative.** Like a ticket title with a little context.
2. **Vary the domain** — backend, frontend, mobile, data, ML, infra, docs, support, billing.
3. **Vary the scale** — some mention "3 files", others "the whole service", others a specific count.
4. **About 1 in 8 in Japanese.** The real workload is bilingual.
5. **No routing hints.** Never write "this is simple", "send this to a big model", "low risk",
   or name any model. The whole point is that Conductor infers difficulty from the task itself.
6. **No duplicates**, and don't reuse the same opening verb more than ~5 times per class.

## Good vs bad

```
GOOD  {"task_class":"debug_hard","task":"Checkout succeeds but the order never appears in the warehouse queue, roughly twice a week."}
GOOD  {"task_class":"extract","task":"請求書PDFから日付・金額・取引先を抽出してください。"}
BAD   {"task_class":"debug_hard","task":"Fix a hard bug."}                    (no substance)
BAD   {"task_class":"summarize","task":"Simple summarisation, use a cheap model."}  (routing hint)
```

## Done when

- `wc -l /Users/soliva/conductor-taskgen/tasks.jsonl` is ≥ 2,200
- every line parses as JSON with exactly `task_class` and `task`
- every `task_class` is one of the 22 above
- each class has ≥ 80 lines
- ≥ 10% of lines contain Japanese characters

Self-check before reporting done:

```bash
python3 - <<'EOF'
import json, collections, re
p = "/Users/soliva/conductor-taskgen/tasks.jsonl"
ok = collections.Counter(); bad = 0; jp = 0
seen = set()
CLASSES = {"summarize","classify","extract","translate","rewrite","commit_message",
 "changelog","mock_data","batch_map","draft_doc","pr_description","boilerplate_code",
 "mechanical_refactor","light_analysis","second_opinion_review","implement_feature",
 "debug_hard","architecture_design","security_review","math_reasoning","multi_step_plan",
 "user_facing_critical"}
for line in open(p, encoding="utf-8"):
    try:
        d = json.loads(line)
        assert set(d) == {"task_class","task"} and d["task_class"] in CLASSES and d["task"].strip()
    except Exception:
        bad += 1; continue
    if d["task"] in seen: bad += 1; continue
    seen.add(d["task"]); ok[d["task_class"]] += 1
    if re.search(r"[぀-ヿ一-鿿]", d["task"]): jp += 1
total = sum(ok.values())
print(f"total {total} · bad/dupe {bad} · japanese {100*jp/max(total,1):.0f}%")
print("min per class:", min(ok.values()) if len(ok)==22 else f"MISSING {sorted(CLASSES-set(ok))}")
EOF
```

Report that output. If `bad` is more than ~2% or any class is under 80, top it up and re-run.

## Do not

- Do not decide routing, tiers, or which model should handle anything.
- Do not touch any git repo, and do not commit or push.
- Do not write anywhere except `/Users/soliva/conductor-taskgen/`.
- Do not install packages or change services.

## Where this goes afterwards (for John, not for you)

Repo: https://github.com/jonpol01/conductor-lm — the file gets wired into
`datagen/generate.py`, replacing the hardcoded `T` template dict. Labels keep coming from the
rule oracle in `datagen/policy.py`, so Grok never influences the routing answer, only the
variety of the questions.
