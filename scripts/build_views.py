"""Generate derived views for the Hospital Deterioration Dataset.

Outputs (by default into ./generated):
- hospital_deterioration_hourly_panel.csv
- hospital_deterioration_ml_ready.csv

These files can be large, so they are not committed to git.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _compute_deterioration_next_12h(hour: pd.Series, det_hour: pd.Series) -> pd.Series:
    """Label: 1 if t < deterioration_hour <= t + 12; else 0."""
    valid_event = det_hour >= 0
    label = valid_event & (det_hour > hour) & (det_hour <= (hour + 12))
    return label.astype(int)


def build(data_dir: Path, out_dir: Path, max_patients: int | None) -> None:
    patients = pd.read_csv(data_dir / "patients.csv")
    vitals = pd.read_csv(data_dir / "vitals_timeseries.csv")
    labs = pd.read_csv(data_dir / "labs_timeseries.csv")

    if max_patients is not None:
        keep = set(patients["patient_id"].head(max_patients).tolist())
        patients = patients[patients["patient_id"].isin(keep)].copy()
        vitals = vitals[vitals["patient_id"].isin(keep)].copy()
        labs = labs[labs["patient_id"].isin(keep)].copy()

    panel = (
        vitals.merge(labs, on=["patient_id", "hour_from_admission"], how="inner")
        .merge(patients, on="patient_id", how="left")
        .sort_values(["patient_id", "hour_from_admission"])
        .reset_index(drop=True)
    )

    panel["deterioration_next_12h"] = _compute_deterioration_next_12h(
        panel["hour_from_admission"],
        panel["deterioration_hour"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = out_dir / "hospital_deterioration_hourly_panel.csv"
    panel.to_csv(panel_path, index=False)

    feature_cols = [
        "hour_from_admission",
        "heart_rate",
        "respiratory_rate",
        "spo2_pct",
        "temperature_c",
        "systolic_bp",
        "diastolic_bp",
        "oxygen_device",
        "oxygen_flow",
        "mobility_score",
        "nurse_alert",
        "wbc_count",
        "lactate",
        "creatinine",
        "crp_level",
        "hemoglobin",
        "sepsis_risk_score",
        "age",
        "gender",
        "comorbidity_index",
        "admission_type",
    ]

    ml = panel[feature_cols + ["deterioration_next_12h"]].copy()

    ml_path = out_dir / "hospital_deterioration_ml_ready.csv"
    ml.to_csv(ml_path, index=False)

    print("✅ Wrote:")
    print(f"  - {panel_path.as_posix()}")
    print(f"  - {ml_path.as_posix()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build derived views for this dataset")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("generated"))
    p.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional: only build for the first N patients (useful for quick tests)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build(args.data_dir, args.out_dir, args.max_patients)


if __name__ == "__main__":
    main()
