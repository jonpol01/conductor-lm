"""Keep Conductor resident and answer routing requests over HTTP.

Loading the base model costs >2 minutes, which is fine for a batch eval and absurd
for a router that is supposed to decide in under 500ms. This loads once and stays up.

  python serve/server.py --adapter runs/e2b-v0/final --port 8770

  curl -s localhost:8770/route -d '{"task":"Summarise this log","context":{"tokens_est":40000,
       "domain":"devops","criticality":"normal","flags":[]},"budget":{"frontier_tokens_remaining":0.5},
       "history":[]}' | jq

Endpoints:
  GET  /health    uptime, request count, latency p50/p95
  POST /route     envelope -> decision only (advisor mode; you dispatch)
  POST /complete  {"prompt": "..."} -> Conductor picks a tier, CALLS it, returns the
                  answer. Gateway mode: you talk to one endpoint and never think about
                  routing. Every decision + outcome is appended to logs/sessions.jsonl,
                  which is the Stage-1/Stage-2 training data the pipeline is missing.

Executors are configured in EXECUTORS below. Any tier without credentials degrades to
"decided but not dispatched" rather than failing the request.
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from jsonschema import Draft7Validator
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from common import load_base  # noqa: E402
from serve.try_decision import DEFAULT_FLEET  # noqa: E402


def extract_json(text):
    """First balanced {...} object in the reply, or None."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`\n")
        if t.startswith("json"):
            t = t[4:]
    i = t.find("{")
    if i < 0:
        return None
    depth = 0
    for j, ch in enumerate(t[i:], i):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None

STATE = {"lat": [], "n": 0, "t0": time.time()}
SESSION_LOG = os.path.join(ROOT, "logs", "sessions.jsonl")

# Where each tier actually runs. The local tier is LM Studio on Hermes; hosted tiers
# need a key and are skipped (decision returned, not executed) when one is absent.
EXECUTORS = {
    "local-2b":      {"url": "http://192.168.2.55:1234/v1/chat/completions",
                      "model": "gemma-4-e2b-it-mlx", "key_env": None},
    "gemma-e2b-local": {"url": "http://192.168.2.55:1234/v1/chat/completions",
                        "model": "gemma-4-e2b-it-mlx", "key_env": None},
    "local-8b":      {"url": "http://192.168.2.55:1234/v1/chat/completions",
                      "model": "qwen3.5-9b-mlx", "key_env": None},
    "claude-sonnet": {"url": "https://api.anthropic.com/v1/messages",
                      "model": "claude-sonnet-5", "key_env": "ANTHROPIC_API_KEY"},
    "claude-opus":   {"url": "https://api.anthropic.com/v1/messages",
                      "model": "claude-opus-5", "key_env": "ANTHROPIC_API_KEY"},
}


def dispatch(route, prompt, timeout=180):
    """Send the REAL prompt to the chosen executor. Returns (text, error)."""
    import urllib.error
    import urllib.request
    ex = EXECUTORS.get(route)
    if ex is None:
        return None, f"no executor configured for {route}"
    key = os.environ.get(ex["key_env"]) if ex["key_env"] else None
    if ex["key_env"] and not key:
        return None, f"{ex['key_env']} not set — decided but not dispatched"
    if "anthropic" in ex["url"]:
        body = {"model": ex["model"], "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]}
        hdrs = {"content-type": "application/json", "x-api-key": key,
                "anthropic-version": "2023-06-01"}
        pick = lambda j: "".join(b.get("text", "") for b in j.get("content", []))  # noqa: E731
    else:
        body = {"model": ex["model"], "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]}
        hdrs = {"content-type": "application/json"}
        pick = lambda j: j["choices"][0]["message"]["content"]  # noqa: E731
    req = urllib.request.Request(ex["url"], data=json.dumps(body).encode(), headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return pick(json.load(r)), None
    except Exception as e:                                       # noqa: BLE001
        return None, repr(e)[:200]


def log_session(rec):
    """One line per completed request — the Stage-1/2 corpus, written as it happens."""
    os.makedirs(os.path.dirname(SESSION_LOG), exist_ok=True)
    with open(SESSION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build(model, tok, sys_prompt, validator, envelope, max_new, gateway=False):
    # gateway mode takes a raw prompt and derives the envelope from it — the caller
    # should not have to hand-build routing metadata just to ask a question
    prompt = envelope.pop("prompt", None)
    if prompt is not None:
        envelope.setdefault("task", prompt[:300])
        envelope.setdefault("context", {})
        envelope["context"].setdefault("tokens_est", max(1, len(prompt) // 4))
    if "fleet" not in envelope:
        envelope["fleet"] = DEFAULT_FLEET
    envelope.setdefault("history", [])
    envelope.setdefault("budget", {"frontier_tokens_remaining": 0.5})
    ctx = envelope.setdefault("context", {})
    ctx.setdefault("tokens_est", 1000)
    ctx.setdefault("domain", "eng")
    ctx.setdefault("criticality", "normal")
    ctx.setdefault("flags", [])

    msgs = [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(envelope, ensure_ascii=False)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    ms = 1000 * (time.time() - t0)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    d = extract_json(text)
    errs = [] if d is None else [e.message for e in validator.iter_errors(d)]
    STATE["lat"].append(ms)
    STATE["n"] += 1
    out = {"decision": d, "raw": None if d else text[:200],
           "schema_errors": errs, "latency_ms": round(ms, 1),
           "envelope": envelope}

    if gateway and prompt is not None:
        route = (d or {}).get("route")
        # fail up: an unusable decision must not silently become a cheap one
        if not route or errs:
            route = "claude-opus"
            out["guard"] = "decision unusable — failed up to frontier"
        answer, err = dispatch(route, prompt)
        out.update({"executed_by": route, "answer": answer, "exec_error": err})
        log_session({"ts": time.time(), "prompt": prompt[:2000],
                     "envelope": envelope, "decision": d,
                     "schema_errors": errs, "route_used": route,
                     "ok": err is None, "error": err,
                     "decision_ms": round(ms, 1)})
    return out


def make_handler(fn):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/health"):
                lat = sorted(STATE["lat"])
                self._send(200, {
                    "ok": True, "requests": STATE["n"],
                    "uptime_s": round(time.time() - STATE["t0"], 1),
                    "p50_ms": round(lat[len(lat) // 2], 1) if lat else None,
                    "p95_ms": round(lat[int(len(lat) * 0.95)], 1) if lat else None})
            else:
                self._send(404, {"error": "GET /health or POST /route"})

        def do_POST(self):
            if not (self.path.startswith("/route") or self.path.startswith("/complete")):
                return self._send(404, {"error": "POST /route or /complete"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                env = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, {"error": f"bad json: {e}"})
            try:
                self._send(200, fn(env, self.path.startswith("/complete")))
            except Exception as e:                       # noqa: BLE001
                self._send(500, {"error": repr(e)})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--adapter", default=os.path.join(ROOT, "runs", "e2b-v0", "final"))
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--max-new", type=int, default=160, dest="max_new")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "spec", "decision.schema.json"), encoding="utf-8") as f:
        validator = Draft7Validator(json.load(f))
    with open(os.path.join(ROOT, "spec", "system_prompt.txt"), encoding="utf-8") as f:
        sys_prompt = f.read().strip()

    print("loading…", flush=True)
    model, tok = load_base(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"ready on :{args.port} (adapter={os.path.basename(args.adapter)})", flush=True)

    srv = ThreadingHTTPServer(
        ("0.0.0.0", args.port),
        make_handler(lambda env, gw=False: build(model, tok, sys_prompt, validator,
                                                 env, args.max_new, gw)))
    srv.serve_forever()


if __name__ == "__main__":
    main()
