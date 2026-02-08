"""Validate the Hospital Deterioration Dataset (synthetic).

This repository stores three canonical CSVs:
- data/patients.csv
- data/vitals_timeseries.csv
- data/labs_timeseries.csv

The README and data dictionary also describe two derived views
(hourly joined panel + ML-ready view). Those are intentionally not committed
because they can be large; you can generate them via scripts/build_views.py.

This script does not modify any dataset files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _fail(message: str) -> None:
    print(f"❌ {message}", file=sys.stderr)
    raise SystemExit(1)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _assert_columns(df: pd.DataFrame, required: set[str], table: str) -> None:
    missing = sorted(required - set(df.columns))
    _assert(not missing, f"{table}: missing columns: {missing}")


def _assert_unique(df: pd.DataFrame, cols: list[str], table: str) -> None:
    dup = int(df.duplicated(subset=cols).sum())
    _assert(dup == 0, f"{table}: {dup} duplicate rows for key {cols}")


def _assert_in_set(series: pd.Series, allowed: set, name: str) -> None:
    invalid = series[~series.isin(allowed)]
    if not invalid.empty:
        sample = invalid.head(10).tolist()
        _fail(f"{name}: found values outside {sorted(allowed)}. Sample: {sample}")


def _assert_between(series: pd.Series, lo: float, hi: float, name: str) -> None:
    bad = series[(series < lo) | (series > hi)]
    if not bad.empty:
        sample = bad.head(10).tolist()
        _fail(f"{name}: values out of range [{lo}, {hi}]. Sample: {sample}")


def _assert_fk(
    child: pd.DataFrame,
    child_col: str,
    parent: pd.DataFrame,
    parent_col: str,
    child_table: str,
    parent_table: str,
) -> None:
    missing = set(child[child_col].dropna().unique()) - set(parent[parent_col].dropna().unique())
    if missing:
        sample = sorted(list(missing))[:10]
        _fail(
            f"{child_table}.{child_col} has {len(missing)} values not present in "
            f"{parent_table}.{parent_col}. Sample: {sample}"
        )


def validate(data_dir: Path, *, strict: bool = False) -> None:
    patients_path = data_dir / "patients.csv"
    vitals_path = data_dir / "vitals_timeseries.csv"
    labs_path = data_dir / "labs_timeseries.csv"

    for p in [patients_path, vitals_path, labs_path]:
        _assert(p.exists(), f"Missing required file: {p.as_posix()}")

    patients = pd.read_csv(patients_path)
    vitals = pd.read_csv(vitals_path)
    labs = pd.read_csv(labs_path)

    # ----- patients -----
    _assert_columns(
        patients,
        {
            "patient_id",
            "age",
            "gender",
            "comorbidity_index",
            "admission_type",
            "baseline_risk_score",
            "los_hours",
            "deterioration_event",
            "deterioration_within_12h_from_admission",
            "deterioration_hour",
        },
        "patients.csv",
    )
    _assert_unique(patients, ["patient_id"], "patients.csv")

    _assert_between(patients["age"], 18, 90, "patients.age")
    _assert_in_set(patients["gender"], {"M", "F"}, "patients.gender")
    _assert_between(patients["comorbidity_index"], 0, 8, "patients.comorbidity_index")
    _assert_in_set(
        patients["admission_type"],
        {"ED", "Elective", "Transfer"},
        "patients.admission_type",
    )
    _assert_between(patients["baseline_risk_score"], 0.0, 1.0, "patients.baseline_risk_score")
    _assert_between(patients["los_hours"], 12, 72, "patients.los_hours")

    _assert_in_set(patients["deterioration_event"], {0, 1}, "patients.deterioration_event")
    _assert_in_set(
        patients["deterioration_within_12h_from_admission"],
        {0, 1},
        "patients.deterioration_within_12h_from_admission",
    )

    event_mask = patients["deterioration_event"] == 1
    no_event_mask = ~event_mask

    _assert(
        (patients.loc[no_event_mask, "deterioration_hour"] == -1).all(),
        "patients.deterioration_hour must be -1 when deterioration_event == 0",
    )

    if event_mask.any():
        hours = patients.loc[event_mask, "deterioration_hour"]
        los = patients.loc[event_mask, "los_hours"]
        _assert((hours >= 0).all(), "patients.deterioration_hour must be >= 0 when event == 1")
        _assert(
            (hours < los).all(),
            "patients.deterioration_hour must be < los_hours when event == 1",
        )

    within_mask = patients["deterioration_within_12h_from_admission"] == 1
    if within_mask.any():
        _assert(
            (patients.loc[within_mask, "deterioration_event"] == 1).all(),
            "within_12h_from_admission==1 requires deterioration_event==1",
        )
        _assert(
            (patients.loc[within_mask, "deterioration_hour"].between(0, 12)).all(),
            "within_12h_from_admission==1 requires deterioration_hour in [0, 12]",
        )

    implied_within = event_mask & patients["deterioration_hour"].between(0, 12)
    if implied_within.any():
        _assert(
            (patients.loc[implied_within, "deterioration_within_12h_from_admission"] == 1).all(),
            "If event hour is within first 12h, within_12h_from_admission should be 1",
        )

    # ----- vitals -----
    _assert_columns(
        vitals,
        {
            "patient_id",
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
        },
        "vitals_timeseries.csv",
    )
    _assert_unique(vitals, ["patient_id", "hour_from_admission"], "vitals_timeseries.csv")
    _assert_fk(
        vitals,
        "patient_id",
        patients,
        "patient_id",
        "vitals_timeseries.csv",
        "patients.csv",
    )

    _assert_between(vitals["hour_from_admission"], 0, 71, "vitals.hour_from_admission")
    _assert_between(vitals["heart_rate"], 20, 250, "vitals.heart_rate")
    _assert_between(vitals["respiratory_rate"], 4, 80, "vitals.respiratory_rate")
    _assert_between(vitals["spo2_pct"], 0, 100, "vitals.spo2_pct")
    _assert_between(vitals["temperature_c"], 30, 45, "vitals.temperature_c")
    _assert_between(vitals["systolic_bp"], 40, 250, "vitals.systolic_bp")
    _assert_between(vitals["diastolic_bp"], 20, 200, "vitals.diastolic_bp")

    _assert_in_set(
        vitals["oxygen_device"],
        {"none", "nasal", "mask", "hfnc", "niv"},
        "vitals.oxygen_device",
    )
    _assert((vitals["oxygen_flow"] >= 0).all(), "vitals.oxygen_flow must be >= 0")

    none_mask = vitals["oxygen_device"] == "none"
    _assert(
        (vitals.loc[none_mask, "oxygen_flow"] == 0.0).all(),
        "vitals: oxygen_flow must be 0.0 when oxygen_device == 'none'",
    )

    # Soft integrity check:
    # Some rows have oxygen_device != 'none' but oxygen_flow == 0.0.
    # We treat those as "unknown / missing flow" by default, but you can fail hard with --strict.
    on_mask = ~none_mask
    if on_mask.any():
        bad = vitals.loc[
            on_mask & (vitals["oxygen_flow"] <= 0.0),
            ["patient_id", "hour_from_admission", "oxygen_device", "oxygen_flow"],
        ]
        if len(bad) > 0:
            sample = bad.head(5).to_dict(orient="records")
            msg = (
                "vitals: oxygen_flow should usually be > 0 when oxygen_device != 'none'. "
                f"Found {len(bad)} rows with oxygen_flow <= 0.0. Sample: {sample}"
            )
            if strict:
                _assert(False, msg)
            else:
                print(f"⚠️  {msg}")

    _assert_between(vitals["mobility_score"], 0, 4, "vitals.mobility_score")
    _assert_in_set(vitals["nurse_alert"], {0, 1}, "vitals.nurse_alert")

    # ----- labs -----
    _assert_columns(
        labs,
        {
            "patient_id",
            "hour_from_admission",
            "wbc_count",
            "lactate",
            "creatinine",
            "crp_level",
            "hemoglobin",
            "sepsis_risk_score",
        },
        "labs_timeseries.csv",
    )
    _assert_unique(labs, ["patient_id", "hour_from_admission"], "labs_timeseries.csv")
    _assert_fk(labs, "patient_id", patients, "patient_id", "labs_timeseries.csv", "patients.csv")

    _assert_between(labs["hour_from_admission"], 0, 71, "labs.hour_from_admission")
    _assert_between(labs["wbc_count"], 0, 100, "labs.wbc_count")
    _assert_between(labs["lactate"], 0, 50, "labs.lactate")
    _assert_between(labs["creatinine"], 0, 50, "labs.creatinine")
    _assert_between(labs["crp_level"], 0, 1000, "labs.crp_level")
    _assert_between(labs["hemoglobin"], 0, 30, "labs.hemoglobin")
    _assert_between(labs["sepsis_risk_score"], 0.0, 1.0, "labs.sepsis_risk_score")

    # ----- alignment: vitals vs labs -----
    keys_v = vitals[["patient_id", "hour_from_admission"]]
    keys_l = labs[["patient_id", "hour_from_admission"]]
    merged = keys_v.merge(
        keys_l,
        on=["patient_id", "hour_from_admission"],
        how="outer",
        indicator=True,
    )

    left_only = int((merged["_merge"] == "left_only").sum())
    right_only = int((merged["_merge"] == "right_only").sum())
    _assert(left_only == 0 and right_only == 0, "vitals/labs keys are not perfectly aligned")

    # ----- per-patient LOS alignment -----
    los_map = patients.set_index("patient_id")["los_hours"]

    v_counts = vitals.groupby("patient_id").size()
    l_counts = labs.groupby("patient_id").size()

    _assert(
        set(v_counts.index) == set(los_map.index),
        "vitals: patient_id set must match patients.csv",
    )
    _assert(
        set(l_counts.index) == set(los_map.index),
        "labs: patient_id set must match patients.csv",
    )

    bad_v = v_counts != los_map.loc[v_counts.index]
    bad_l = l_counts != los_map.loc[l_counts.index]

    _assert(
        not bad_v.any(),
        f"vitals: per-patient row count must equal los_hours. Bad patients: {int(bad_v.sum())}",
    )
    _assert(
        not bad_l.any(),
        f"labs: per-patient row count must equal los_hours. Bad patients: {int(bad_l.sum())}",
    )

    v_min = vitals.groupby("patient_id")["hour_from_admission"].min()
    v_max = vitals.groupby("patient_id")["hour_from_admission"].max()
    expected_max = los_map - 1

    _assert((v_min == 0).all(), "vitals: hour_from_admission must start at 0 for every patient")
    _assert(
        (v_max == expected_max.loc[v_max.index]).all(),
        "vitals: hour_from_admission max must equal los_hours - 1",
    )

    l_min = labs.groupby("patient_id")["hour_from_admission"].min()
    l_max = labs.groupby("patient_id")["hour_from_admission"].max()

    _assert((l_min == 0).all(), "labs: hour_from_admission must start at 0 for every patient")
    _assert(
        (l_max == expected_max.loc[l_max.index]).all(),
        "labs: hour_from_admission max must equal los_hours - 1",
    )

    print("✅ Dataset validation passed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the Hospital Deterioration Dataset")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to the data directory")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on soft issues (treat warnings as errors)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    validate(args.data_dir, strict=args.strict)


if __name__ == "__main__":
    main()
