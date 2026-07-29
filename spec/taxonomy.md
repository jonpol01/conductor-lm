# Routing taxonomy v0

Task classes, grouped by the tier a default (unconditioned) policy prefers.
Conditioning signals — context size, criticality, flags, fleet registry, budget,
failure history — can override the default per the precedence rules below.

## Task classes

| Group | Classes |
|---|---|
| LOCAL-preferred | `summarize`, `classify`, `extract`, `translate`, `rewrite`, `commit_message`, `changelog`, `mock_data`, `batch_map` |
| MID-preferred | `draft_doc`, `pr_description`, `boilerplate_code`, `mechanical_refactor`, `light_analysis`, `second_opinion_review` |
| FRONTIER-preferred | `implement_feature`, `debug_hard`, `architecture_design`, `security_review`, `math_reasoning`, `multi_step_plan`, `user_facing_critical` |

## Precedence (highest first)

1. **Privacy** — `privacy_required` flag pins the task to the local tier.
2. **Criticality** — `high`/`security` criticality, or any FRONTIER-preferred class → frontier.
3. **Ambiguity** — `ambiguous` flag fails the class-default route up one tier.
4. **History** — a `failed`/`escalated` outcome for the same task class at the target tier routes one tier up.
5. **Capability gates** — context length vs `ctx_max`; task class vs declared `caps`; missing tier in fleet. All gates fail **up**, never down.
6. **Budget** — budget pressure may downroute MID-preferred work to a *capable* local executor; it never downroutes past a capability gate or criticality rule.
7. **Class default** — the group preference above.

## Invariants

- Fail up, never down: no rule may resolve uncertainty toward a cheaper tier.
- Every decision carries exactly one `rationale_class` from the closed vocabulary
  ([rationale_classes.json](rationale_classes.json)) identifying the *binding* rule.
- The `route` and every `plan[].route` / `escalation.to` must be executor ids present
  in the envelope's fleet registry (`escalation.to` may be `null` at the frontier).
