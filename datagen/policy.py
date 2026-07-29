"""Gold routing policy v0 — the deterministic rule oracle for Stage-0 data.

Implements the precedence order in spec/taxonomy.md. Every decision it emits
must validate against spec/decision.schema.json. Deterministic w.r.t. the
envelope (no RNG) so eval replay is exact.
"""

import hashlib
import json

LOCAL_CLASSES = {
    "summarize", "classify", "extract", "translate", "rewrite",
    "commit_message", "changelog", "mock_data", "batch_map",
}
MID_CLASSES = {
    "draft_doc", "pr_description", "boilerplate_code",
    "mechanical_refactor", "light_analysis", "second_opinion_review",
}
FRONTIER_CLASSES = {
    "implement_feature", "debug_hard", "architecture_design",
    "security_review", "math_reasoning", "multi_step_plan",
    "user_facing_critical",
}
CODE_CLASSES = {"boilerplate_code", "mechanical_refactor", "implement_feature", "debug_hard"}

TIER_ORDER = ["local", "mid", "frontier"]


def _by_tier(fleet):
    tiers = {}
    for ex in fleet:
        tiers.setdefault(ex["tier"], []).append(ex)
    return tiers


def _pick(tiers, tier):
    """Cheapest executor of a tier (stable: cost, then id)."""
    xs = tiers.get(tier, [])
    return sorted(xs, key=lambda e: (e["cost"], e["id"]))[0] if xs else None


def _next_up(tiers, tier):
    for t in TIER_ORDER[TIER_ORDER.index(tier) + 1:]:
        ex = _pick(tiers, t)
        if ex:
            return ex
    return None


def _fits(ex, task_class, tokens):
    return task_class in ex["caps"] and tokens <= ex["ctx_max"]


def _jitter(envelope):
    h = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).digest()
    return (h[0] % 7) * 0.01


def _history_bad(envelope, task_class, tier):
    return any(
        h["task_class"] == task_class and h["tier"] == tier
        and h["outcome"] in ("failed", "escalated")
        for h in envelope["history"]
    )


def decide(envelope, task_class):
    """Return (decision dict, binding rule notes). task_class is the generator's
    ground-truth class for the task text (the model must infer it from the text)."""
    ctx = envelope["context"]
    tokens = ctx["tokens_est"]
    flags = set(ctx.get("flags", []))
    crit = ctx.get("criticality", "normal")
    tiers = _by_tier(envelope["fleet"])
    budget = envelope["budget"]["frontier_tokens_remaining"]

    route_ex = None
    rationale = None
    conf = 0.9

    # 1. privacy pins to local (generator guarantees a capable local executor exists)
    if "privacy_required" in flags:
        candidates = [e for e in tiers.get("local", []) if _fits(e, task_class, tokens)]
        route_ex = sorted(candidates, key=lambda e: (e["cost"], e["id"]))[0]
        rationale = "privacy_local_required"

    # 2. criticality / frontier-preferred classes
    elif crit in ("high", "security") or task_class in FRONTIER_CLASSES:
        route_ex = _pick(tiers, "frontier")
        if crit == "security" or task_class == "security_review":
            rationale = "security_sensitive_frontier"
        elif task_class == "architecture_design":
            rationale = "architecture_design_frontier"
        elif task_class in ("math_reasoning", "debug_hard", "multi_step_plan"):
            rationale = "complex_reasoning_frontier"
        else:
            rationale = "correctness_critical_frontier"

    else:
        # class default tier
        default_tier = "local" if task_class in LOCAL_CLASSES else "mid"

        # 3. ambiguity fails the default up one tier
        if "ambiguous" in flags:
            up_tier = TIER_ORDER[TIER_ORDER.index(default_tier) + 1]
            route_ex = _pick(tiers, up_tier) or _next_up(tiers, up_tier)
            rationale = "ambiguity_fail_up"
            conf = 0.6

        elif default_tier == "local":
            local = _pick(tiers, "local")
            if local is None:
                route_ex = _next_up(tiers, "local")
                rationale = "fleet_gap_fail_up"
                conf = 0.75
            elif _history_bad(envelope, task_class, "local"):
                route_ex = _next_up(tiers, "local")
                rationale = "history_failure_escalation"
                conf = 0.75
            elif tokens > local["ctx_max"]:
                mid = _pick(tiers, "mid")
                route_ex = mid if (mid and tokens <= mid["ctx_max"]) else _pick(tiers, "frontier")
                rationale = "long_context_capability_gate"
                conf = 0.75
            elif task_class not in local["caps"]:
                route_ex = _next_up(tiers, "local")
                rationale = "capability_gate_local_unfit"
                conf = 0.75
            else:
                route_ex = local
                rationale = ("high_volume_batch_local"
                             if ("batch" in flags or task_class == "batch_map")
                             else "light_extractive_local")

        else:  # MID-preferred
            mid = _pick(tiers, "mid")
            # 6. budget pressure downroutes ONLY to a capable local executor
            local_capable = [e for e in tiers.get("local", []) if _fits(e, task_class, tokens)]
            if budget < 0.2 and local_capable and not _history_bad(envelope, task_class, "local"):
                route_ex = sorted(local_capable, key=lambda e: (e["cost"], e["id"]))[0]
                rationale = "budget_conservation_downroute"
                conf = 0.75
            elif mid is None:
                route_ex = _pick(tiers, "frontier")
                rationale = "fleet_gap_fail_up"
                conf = 0.75
            elif _history_bad(envelope, task_class, "mid"):
                route_ex = _next_up(tiers, "mid")
                rationale = "history_failure_escalation"
                conf = 0.75
            elif tokens > mid["ctx_max"]:
                route_ex = _pick(tiers, "frontier")
                rationale = "long_context_capability_gate"
                conf = 0.75
            else:
                route_ex = mid
                if task_class == "mechanical_refactor":
                    rationale = "mechanical_refactor_mid"
                elif task_class in CODE_CLASSES:
                    rationale = "low_risk_code_mid"
                else:
                    rationale = "boilerplate_draft_mid"

    # --- plan ---
    route = route_ex["id"]
    plan = [{"step": task_class.replace("_", " "), "route": route}]
    if task_class in ("summarize", "extract") and tokens > 20000:
        plan = [
            {"step": "chunk and " + task_class.replace("_", " "), "route": route},
            {"step": "merge chunk results", "route": route},
        ]
    elif task_class == "mechanical_refactor" and route_ex["tier"] != "frontier":
        up = _next_up(tiers, route_ex["tier"])
        if up:
            plan.append({"step": "review diff", "route": up["id"]})

    # --- escalation ---
    if route_ex["tier"] == "frontier":
        escalation = {"on": [], "to": None}
    else:
        up = _next_up(tiers, route_ex["tier"])
        triggers = (["output_rejected", "test_fail"]
                    if task_class in CODE_CLASSES
                    else ["schema_invalid", "self_reported_uncertainty"])
        escalation = {"on": triggers, "to": up["id"] if up else None}

    return {
        "route": route,
        "plan": plan,
        "escalation": escalation,
        "confidence": round(min(0.99, conf + _jitter(envelope)), 2),
        "rationale_class": rationale,
    }
