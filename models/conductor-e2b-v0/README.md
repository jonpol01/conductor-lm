# Conductor-E2B v0 (Stage-0 adapter)

LoRA adapter for `google/gemma-4-E2B-it` that turns it into a routing controller: it reads a
task envelope and emits a routing decision as constrained JSON. It does not perform tasks.

- **Stage:** 0 (schema grounding + rule imitation). Not outcome-trained.
- **Base:** `google/gemma-4-E2B-it`, loaded in 4-bit nf4
- **Adapter:** LoRA r=16, alpha=32, dropout=0.05, 24.2M trainable params (0.52%)
- **Trained on:** 12,000 synthetic envelopes, 1 epoch, 750 steps, single RTX 3080 (~14 h)
- **Loss:** train 0.0954 / eval 0.0964

## Results (800 held-out envelopes)

| metric | result |
|---|---|
| JSON parse rate | 100.0% |
| Schema validity | 100.0% |
| Route names a real fleet executor | 100.0% |
| Escalation points strictly upward | 100.0% |
| Route agreement with rule oracle | 91.4% |
| Rationale-class agreement | 90.4% |
| Unsafe downroutes | 4.0% (32/800) |

Full breakdown: [../../eval/results/stage0-e2b-v0-800.json](../../eval/results/stage0-e2b-v0-800.json)
and §6.1 of the [root README](../../README.md).

## Known limitation

The fail-up invariant holds for the `escalation` field but **not** for primary route selection.
32 of 800 decisions routed toward a *cheaper* tier than the oracle, and 30 of those come from
the conditional override rules — `history_failure_escalation` alone accounts for 17. In practice
the model reads a history entry saying "this tier already failed for this task class" and routes
there anyway.

**Do not deploy this adapter as a final decision-maker.** Treat its output as a proposal and
enforce the hard gates deterministically in the serving layer.

## Loading — important

`adapter_config.json` stores `target_modules` as seven bare suffixes (`q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), not full paths. PEFT matches those by
suffix, and Gemma 4 E2B's **audio and vision towers contain attention blocks using the same
names**. If the towers are attached when you load the adapter, PEFT tries to inject into them and
fails with `Target module Gemma4ClippableLinear(...) is not supported`.

Strip the towers first. `common.load_base()` in this repo does it for you:

```python
from common import load_base
from peft import PeftModel

model, tok = load_base("google/gemma-4-E2B-it")        # 4-bit + towers removed
model = PeftModel.from_pretrained(model, "models/conductor-e2b-v0")
model.eval()
```

Then prompt with the system prompt from [`spec/system_prompt.txt`](../../spec/system_prompt.txt)
and a JSON envelope conforming to [`spec/envelope.schema.json`](../../spec/envelope.schema.json)
as the user turn. Decode greedily; validate the reply against
[`spec/decision.schema.json`](../../spec/decision.schema.json).

## Files

The tokenizer is deliberately **not** included — it is byte-identical to the base model's, so
load it from `google/gemma-4-E2B-it`. `adapter_model.safetensors` is stored via Git LFS.

## License

MIT for the adapter weights. The base model is governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
