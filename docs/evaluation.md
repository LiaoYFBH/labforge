# Lab-Forge Evaluation

Lab-Forge ships with a **Benchmark-Sampled Mini-Eval** under
[`evals/mini_eval/`](../evals/mini_eval/). It is a lightweight case-study
evaluation, **not** an official leaderboard run, and does not claim
state-of-the-art performance.

## Why a Mini-Eval

The Mini-Eval is sized for an overnight run and a small CPU host:

- It samples **1–3 items** per public source rather than running full
  benchmark splits.
- It uses **deterministic adapters** for the literature-centric cases so
  the eval verifies *plumbing* (dataset attachment, artifact production,
  numeric grounding, forbidden-claim absence) rather than open-ended
  agent intelligence.
- It performs **real scikit-learn computation** for the ML-execution
  case, then validates that the experiment report's numbers come back
  out of `metrics.json`.

## Diagnostic vs Holdout

The suite is split into two case sets so we can tighten reliability
without contaminating the reported results:

| Set | Used for |
|---|---|
| `diagnostic` | Internal debugging; not shown in the README table. |
| `holdout` | Reporting / README. Different sample slice from diagnostic. |

If a case in either set cannot fetch its public dataset within the
budget, it is marked `SKIPPED_OFFLINE_OR_HEAVY` and the
`source_manifest.json` records the reason honestly. There is **no**
synthetic-data fallback.

## Running

```bash
python evals/mini_eval/run_eval.py --set diagnostic
python evals/mini_eval/run_eval.py --set holdout
python evals/mini_eval/run_eval.py --set all
```

Outputs land in `evals/mini_eval/outputs/latest/`. The README table is
refreshed at `evals/mini_eval/outputs/latest/readme_eval_table.md`.

## Validators

[`validators.py`](../evals/mini_eval/validators.py) provides task-agnostic
checks used across cases:

- `validate_required_artifacts` — required output files are present.
- `validate_source_manifest` — `source_manifest.json` is well-formed.
- `validate_forbidden_claims` — text does not contain phrases like
  "state-of-the-art", "official benchmark", "fully evaluated on".
- `validate_numeric_grounding` — every numeric token in a report is
  either a structural integer (sample counts) or matches a value present
  in `metrics.json` within tolerance.
- `validate_citation_ids` — citations refer only to provided source ids.

These validators are deliberately generic. Cases never hardcode
dataset-specific answers (no `if "20news" in ...` branches in the
runner).

## Honest disclaimers

- Lab-Forge Mini-Eval is a lightweight case-study evaluation inspired by
  public scientific-agent benchmarks. It is not an official leaderboard
  evaluation and does not claim state-of-the-art performance.
- Sample sizes are tiny by design.
- HuggingFace SSL handshakes are intermittently flaky in some
  environments; literature-centric cases may legitimately mark
  `SKIPPED_OFFLINE_OR_HEAVY` on those runs. Re-run the eval to refresh.
