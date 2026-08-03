"""Keep Conductor resident and answer routing requests over HTTP.

Loading the base model costs >2 minutes, which is fine for a batch eval and absurd
for a router that is supposed to decide in under 500ms. This loads once and stays up.

  python serve/server.py --adapter runs/e2b-v0/final --port 8770

  curl -s localhost:8770/route -d '{"task":"Summarise this log","context":{"tokens_est":40000,
       "domain":"devops","criticality":"normal","flags":[]},"budget":{"frontier_tokens_remaining":0.5},
       "history":[]}' | jq

Endpoints:
  GET  /health   model id, uptime, request count, latency p50/p95
  POST /route    envelope (fleet optional — a default fleet is filled in) -> decision
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
from serve.try_decision import DEFAULT_FLEET, extract_json  # noqa: E402

STATE = {"lat": [], "n": 0, "t0": time.time()}


def build(model, tok, sys_prompt, validator, envelope, max_new):
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
    return {"decision": d, "raw": None if d else text[:200],
            "schema_errors": errs, "latency_ms": round(ms, 1),
            "envelope": envelope}


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
            if not self.path.startswith("/route"):
                return self._send(404, {"error": "POST /route"})
            n = int(self.headers.get("Content-Length", 0))
            try:
                env = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as e:
                return self._send(400, {"error": f"bad json: {e}"})
            try:
                self._send(200, fn(env))
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
        make_handler(lambda env: build(model, tok, sys_prompt, validator,
                                       env, args.max_new)))
    srv.serve_forever()


if __name__ == "__main__":
    main()
