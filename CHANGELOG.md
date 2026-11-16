# Changelog

## 1.1.0 — work-in-progress checkpoint

Partial code published for continued development. Known issues: STATUS.md.

# Tensor Hub Changelog

## [2026-11-16] Bug Fix #434

### Fixed
- **Critical Bug**: ArgumentError in build_metrics during threshold tuning
  - **Issue**: During `demo_threshold_tuning`, when model predicts zero positive cases (y_pred contains only 0s), generator comprehensions in `build_metrics` become empty
  - **Impact**: Julia throws `ArgumentError: reducing over an empty collection is not allowed`, causing script to crash before completing threshold tuning table
  - **Location**: `phases/02-ml-fundamentals/03-logistic-regression/code/main.jl`

### Changes Made
- Added `init=0` parameter to all `sum()` calls in `build_metrics` function
- Implemented epsilon handling (`eps = 1e-6`) for division operations to prevent division by zero
- Maintained backward compatibility - no API changes
- Added comprehensive test script `test_build_metrics_fix.jl` to verify the fix

### Testing
- Created test script demonstrating both buggy behavior (crashes with ArgumentError) and fixed behavior (handles empty collections gracefully)
- Verified fix correctly computes metrics for edge cases where y_pred contains only zeros

### Acceptance Criteria Met
✅ `build_metrics` function handles empty generators gracefully  
✅ `demo_threshold_tuning` completes full threshold tuning table without crashing  
✅ Metrics computed correctly for all threshold scenarios  
✅ No test failures  

## Repository Structure
- **test_build_metrics_fix.jl** - Test script demonstrating the bug fix
- **PLAN.md** - Project plan (updated)
- **STATUS.md** - Project status (updated)
- **phases/02-ml-fundamentals/03-logistic-regression/code/main.jl** - Contains the fixed build_metrics function

---
Alan Vo · [alanvo.net](https://alanvo.net) · [X: @alanvoasi](https://x.com/alanvoasi)
