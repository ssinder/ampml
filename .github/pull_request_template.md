## What this PR does

<!-- A one-sentence summary. -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (existing results no longer reproduce)
- [ ] Documentation only
- [ ] Infrastructure / CI

## Does this change any reported metric?

<!-- This is the most important question in this template. -->

- [ ] **No** — no number in `results/` changes.
- [ ] **Yes** — numbers changed.

If yes, please provide a before/after table and explain **why** the change is
correct rather than a regression:

| Model | Metric | Before | After |
|-------|--------|--------|-------|
|       |        |        |       |

> Reviewers will re-run `bash run_all.sh --only 99` and compare against
> `results/verification_report.json`. If the audit fails, the PR cannot be merged.

## Checklist

- [ ] `python src/00_check_environment.py` passes
- [ ] `bash run_all.sh --only 99` passes (all 30 checks)
- [ ] `pytest -q` passes
- [ ] `pre-commit run --all-files` passes
- [ ] I did **not** modify any file under `results/` by hand — all numbers there are generated
- [ ] If I changed the data split, feature selection, or hyperparameters, I documented it in `CHANGELOG.md`
- [ ] New logic has tests in `test/`
- [ ] I did not introduce a new random seed inline (seed comes from `src/config.py`)
- [ ] I did not hardcode absolute paths (paths come from `src/config.py`)

## Related issues

<!-- Closes #123 -->
