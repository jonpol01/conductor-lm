"""Talk to a running Conductor server in plain English.

  python serve/ask.py "Summarise this 40k-token incident log" --tokens 41000
  python serve/ask.py --demo
"""

import argparse
import json
import sys
import urllib.request

DEMO = [
    ("Translate these 30 UI strings from Japanese to English", 1200, "normal", []),
    ("Write a commit message for this staged diff", 400, "low", []),
    ("Audit this file-upload endpoint for path traversal", 7000, "security", []),
    ("Summarise these internal HR complaint records", 9000, "normal", ["privacy_required"]),
    ("Plan the zero-downtime Postgres 14->16 upgrade across three environments",
     5000, "high", []),
    ("Rename getUserInfo to fetchUserProfile across the codebase", 30000, "normal", []),
    ("The worker deadlocks under load once a day; find the root cause", 60000, "high", []),
]


def post(url, env, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(env).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def show(task, r):
    d = r.get("decision") or {}
    ok = "ok" if not r.get("schema_errors") else f"INVALID {r['schema_errors'][:1]}"
    print(f"  {task[:64]}")
    if not d:
        print(f"     UNPARSEABLE: {(r.get('raw') or '')[:90]}\n")
        return
    esc = d.get("escalation") or {}
    print(f"     route      {d.get('route')}")
    print(f"     because    {d.get('rationale_class')}")
    print(f"     escalate   {esc.get('to')} on {esc.get('on')}")
    print(f"     confidence {d.get('confidence')}   latency {r.get('latency_ms')}ms   "
          f"schema {ok}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?")
    ap.add_argument("--host", default="http://localhost:8770")
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--criticality", default="normal")
    ap.add_argument("--flags", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    url = args.host.rstrip("/") + "/route"
    if args.demo:
        for task, tok, crit, flags in DEMO:
            show(task, post(url, {"task": task,
                                  "context": {"tokens_est": tok, "criticality": crit,
                                              "flags": flags}}))
        print(json.dumps(json.load(urllib.request.urlopen(
            args.host.rstrip("/") + "/health")), indent=1))
        return
    if not args.task:
        sys.exit("give a task, or --demo")
    show(args.task, post(url, {
        "task": args.task,
        "context": {"tokens_est": args.tokens, "criticality": args.criticality,
                    "flags": [f for f in args.flags.split(",") if f]}}))


if __name__ == "__main__":
    main()
