# Hermes cron: feed Conductor real traffic and collect its decisions

**What this is.** Conductor is a small router running on the 3080. It reads a task, picks which
tier of model should handle it, calls that model, and returns the answer. Every call it serves
is one row of training data the project currently does not have.

**Your job.** Run a scheduled job that sends real work through it and records what happened.
You are not training anything and not judging its routing — you are generating traffic.

---

## The endpoint

```
http://192.168.2.109:8770/complete     POST {"prompt": "..."}  -> answer, routed automatically
http://192.168.2.109:8770/health       GET   liveness + latency
```

It lives on the RTX 3080 box (WSL). If `/health` does not answer, **skip the run and report** —
do not try to start it, do not SSH into the box, do not install anything.

A decision currently takes ~14s, and the local tier then takes a few more. Budget ~60s per
prompt and keep batches small.

## Install

Create `/Users/soliva/conductor-cron/run.sh` with exactly this:

```bash
#!/bin/bash
# Feed Conductor a small batch of real tasks and log what it decided.
set -u
HOST=http://192.168.2.109:8770
OUT=/Users/soliva/conductor-cron
mkdir -p "$OUT"
LOG="$OUT/runs.log"
DATA="$OUT/local-sessions.jsonl"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# 1. guard: never queue work at a server that is down
if ! curl -sf --max-time 10 "$HOST/health" >/dev/null; then
  echo "$(stamp) SKIP conductor down" >> "$LOG"; exit 0
fi

# 2. prompt source, in order of preference:
#    a) /Users/soliva/conductor-cron/queue.txt  — one real task per line, consumed
#    b) the built-in pool below, when the queue is empty
BATCH=8
PROMPTS=()
if [ -s "$OUT/queue.txt" ]; then
  while IFS= read -r l; do [ -n "$l" ] && PROMPTS+=("$l"); done < <(head -n "$BATCH" "$OUT/queue.txt")
  tail -n +$((BATCH+1)) "$OUT/queue.txt" > "$OUT/queue.tmp" && mv "$OUT/queue.tmp" "$OUT/queue.txt"
fi
if [ ${#PROMPTS[@]} -eq 0 ]; then
  while IFS= read -r l; do PROMPTS+=("$l"); done < <(shuf -n "$BATCH" "$OUT/pool.txt")
fi

# 3. send them one at a time; record decision + outcome
ok=0; fail=0
for p in "${PROMPTS[@]}"; do
  body=$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1]}))' "$p")
  resp=$(curl -s --max-time 180 -X POST "$HOST/complete" \
           -H 'content-type: application/json' -d "$body")
  line=$(python3 - "$resp" "$p" <<'PY'
import json, sys, time
raw, prompt = sys.argv[1], sys.argv[2]
try:
    r = json.loads(raw)
except Exception:
    print(json.dumps({"ts": time.time(), "prompt": prompt[:300], "error": "unparseable response"}))
    sys.exit(0)
d = r.get("decision") or {}
print(json.dumps({
    "ts": time.time(), "prompt": prompt[:300],
    "route": d.get("route"), "rationale": d.get("rationale_class"),
    "confidence": d.get("confidence"),
    "schema_errors": r.get("schema_errors") or [],
    "executed_by": r.get("executed_by"),
    "ok": bool(r.get("answer")),
    "error": r.get("exec_error"),
    "answer_chars": len(r.get("answer") or ""),
    "decision_ms": r.get("latency_ms"),
}, ensure_ascii=False))
PY
)
  echo "$line" >> "$DATA"
  echo "$line" | grep -q '"ok": true' && ok=$((ok+1)) || fail=$((fail+1))
done

echo "$(stamp) sent=${#PROMPTS[@]} ok=$ok fail=$fail" >> "$LOG"
```

Then:

```bash
chmod +x /Users/soliva/conductor-cron/run.sh
```

## The built-in pool

Create `/Users/soliva/conductor-cron/pool.txt`, one task per line. Aim for 60+ lines covering a
realistic spread of work — light extraction, routine drafting, code, and genuinely hard
problems. A few in Japanese. **Do not write anything about difficulty or which model should
handle it** — inferring that is the whole point of the thing you are testing. Starter lines:

```
Summarise this week's deploy log into the recurring error signatures.
Write a conventional-commit message for a change that adds retry-with-backoff to the webhook handler.
Extract every endpoint and its auth requirement from this OpenAPI spec.
このリリースノートを日本語に翻訳してください。
Rename getUserInfo to fetchUserProfile across the codebase and fix the imports.
Draft an internal FAQ page for the new deployment process.
Diagnose why the worker deadlocks under load roughly once a day.
Design the module boundaries for splitting this monolith, with trade-offs.
Review this file-upload endpoint for path traversal and content-type bypasses.
Plan a zero-downtime Postgres 14 to 16 upgrade across three environments.
```

## Preferred: use real work

If you have actual task descriptions from your own queue, append them to
`/Users/soliva/conductor-cron/queue.txt`, one per line. The script drains that first and only
falls back to the pool when it is empty. **Real tasks are worth far more than invented ones** —
the whole reason this job exists is that synthetic tasks are all the project has so far.

Strip anything sensitive first: no credentials, no customer names, no personal data. Conductor
only needs the shape of the task, never its payload.

## Cron

Every 3 hours, staggered off the hour so it does not collide with your other jobs:

```
17 */3 * * * /bin/bash /Users/soliva/conductor-cron/run.sh
```

Add with `crontab -e`. Confirm it landed with `crontab -l | grep conductor`.

## Report

After the first manual run, and then only when something looks wrong, post to Discord:

- `tail -3 /Users/soliva/conductor-cron/runs.log`
- the count so far: `wc -l /Users/soliva/conductor-cron/local-sessions.jsonl`
- flag it if `fail` exceeds `ok` on two consecutive runs, or if you see `SKIP conductor down`
  three runs in a row

Do not post every run — this is a background collector, not an alerting system.

## First run: do this once by hand

```bash
/bin/bash /Users/soliva/conductor-cron/run.sh
tail -2 /Users/soliva/conductor-cron/runs.log
tail -1 /Users/soliva/conductor-cron/local-sessions.jsonl
```

Expect a line like `sent=8 ok=6 fail=2`. Some `fail` is normal and expected — tasks routed to
the frontier tier report *"ANTHROPIC_API_KEY not set — decided but not dispatched"*, which is
still a useful row: the decision was recorded even though nothing executed. What is **not**
normal is `ok=0`, or `unparseable response` on every line.

## Do not

- Do not SSH to the 3080, restart the server, or install packages.
- Do not edit any git repository.
- Do not write outside `/Users/soliva/conductor-cron/`.
- Do not judge or override Conductor's routing — record it, that is all.
- Use absolute `/Users/soliva/...` paths everywhere; `~` resolves to `/Users/hermes` for you.

---

*Context for John, not for the agent:* rows land in `/Users/soliva/conductor-cron/local-sessions.jsonl`
here and in `logs/sessions.jsonl` on the 3080. Together they are the Stage-1 (envelope→decision)
and Stage-2 (outcome) corpora that `docs/rl-plan`-equivalent §5 of the
[README](https://github.com/jonpol01/conductor-lm) says are missing. Frontier rows stay
decision-only until `ANTHROPIC_API_KEY` is set in the server's environment.
