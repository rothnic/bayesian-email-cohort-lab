# Before fixed calendar-grid correction

These scores already use the corrected zero-count variance and direct cheap-source targets. They predate the independent finding that inserting a cutoff-specific knot changes the prior covariance for the same historical date. The final model uses fixed global weekly knots and consistent interpolation in both historical inference and future prediction. No scale-support probabilities were changed.

The previously verified model SHA256 before the two subsequently added input validation guards was15298c8b060bce0430deff587c311c6bfa25d772a41ba986f25ebb29a68deb32. The binary-payout and per-row-negative guards left all existing output CSV bytes unchanged. The complete first-version source is retained in ../pre_half_count_fix/model_before.txt; this intermediate code was not separately archived. Final tests preserve the exact historical/prior covariance and posterior invariance checks.

These are superseded results, retained as review evidence. Use ../../artifacts/run_manifest.json to identify the final result version.
