# Product development plan

Each round completes one feature or repair. Future items are planned, not implemented claims.

## Current round

logistic-regression-demo-bugfix

## Roadmap

### logistic-regression-demo-bugfix — Logistic Regression demo with safe build_metrics

Status: done

Acceptance:
- Demo script runs without errors
- Edge-case predictions (all zeros) produce correct metrics without crash
- Output includes accuracy, precision, recall, f1 for test case

Implementation: `src/tensorhub/demo_logreg.py`

Tests: `tests/demo_logreg_test.py`

---

Alan Vo · [alanvo.net](https://alanvo.net) · [X: @alanvoasi](https://x.com/alanvoasi)
