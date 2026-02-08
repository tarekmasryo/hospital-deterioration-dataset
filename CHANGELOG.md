# Changelog
## v1.0.2 (validator tuning)

**No dataset values changed.**

- Validator now treats a small set of oxygen support edge-cases as **warnings** by default.
- Added `--strict` flag to fail on soft issues when you want a hard gate.


## v1.0.1 (repo hardening)

**No dataset values changed.** This release is focused on making the repository easier to trust, validate, and reuse.

- Added `scripts/validate_dataset.py` for integrity checks (keys, ranges, time-series alignment).
- Added `scripts/build_views.py` to generate the two derived views described in the docs.
- Clarified in `README.md` that derived views live under `generated/` and are not committed.
- Promoted `data_dictionary.md` to the repo root for easier discovery.
- Added `checksums.sha256` so users can verify file integrity.
- Added Ruff config (`pyproject.toml`) + dev requirements.
- Added GitHub Actions workflow for data-quality checks.
