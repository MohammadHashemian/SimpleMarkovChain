"""Tests for the Iran mortality calculator (polars-based)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from notebook_tools.iran_mortality import (
    POLAND_AGE_BUCKETS,
    WPP_AGE_START_TO_POLAND,
    WPP_OPEN_AGE_STARTS,
    aggregate_to_poland_buckets,
    build_mortality_from_wpp,
    build_mortality_table,
    compare_to_reference,
    compute_cohort_mortality,
    compute_crude_annual_rate,
    load_iran_population,
    load_wpp_mortality,
    merge_cohort_with_wpp,
    validate_mortality_table,
    write_mortality_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRAN_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_population(tmp_path: Path) -> tuple[Path, Path]:
    """Two CSVs with a known, deterministic survival pattern.

    Ages 0..100, sexes M and F. For each age x the population drops by
    10% per year (i.e. q(x) = 0.1) so we can verify the cohort method
    and aggregation end-to-end. ``pop_t1`` is constructed by applying
    the cohort mapping: ``pop_t1[x+1] = round(0.9 * pop_t[x])``, so the
    cohort method yields exactly 0.1 at every single-year age.
    """
    ages = list(range(0, 101))
    base = 100_000

    pop_t = {a: max(int(round(base * 0.9**a)), 100) for a in ages}
    pop_t1: dict[int, int] = {}
    pop_t1[0] = max(int(round(0.9 * pop_t[0])), 1)
    for x in range(1, 100):
        pop_t1[x] = max(int(round(0.9 * pop_t[x - 1])), 1)
    pop_t1[100] = max(int(round(0.9 * pop_t[99])), 1)

    def _frame(pop: dict[int, int]) -> pl.DataFrame:
        rows = []
        for a, n in pop.items():
            age_str = "100+" if a == 100 else str(a)
            half = n // 2
            rows.append({"Age": age_str, "M": half, "F": n - half})
        return pl.DataFrame(rows)

    path_t = tmp_path / "Iran_2023.csv"
    path_t1 = tmp_path / "Iran_2024.csv"
    _frame(pop_t).write_csv(path_t)
    _frame(pop_t1).write_csv(path_t1)
    return path_t, path_t1


# Synthetic WPP rates that increase with age (realistic for an adult male).
_SYNTHETIC_RATES = {
    0: 0.010,
    1: 0.0003,
    5: 0.0002,
    10: 0.0002,
    15: 0.0006,
    20: 0.0010,
    25: 0.0013,
    30: 0.0017,
    35: 0.0022,
    40: 0.0030,
    45: 0.0045,
    50: 0.0067,
    55: 0.0100,
    60: 0.0150,
    65: 0.0220,
    70: 0.0340,
    75: 0.0550,
    80: 0.0850,
    85: 0.1400,
    90: 0.2200,
    95: 0.3000,
    100: 0.4000,
}


@pytest.fixture
def synthetic_wpp_csv(tmp_path: Path) -> Path:
    """A UN WPP-shaped CSV with all 22 abridged age groups, Male, 2024."""
    rows = []
    for age_start, rate in _SYNTHETIC_RATES.items():
        if age_start == 0:
            age_end, age_label = 1, "0"
        elif age_start == 1:
            age_end, age_label = 5, "1-4"
        elif age_start == 100:
            age_end, age_label = 100, "100+"
        else:
            age_end = age_start + 4
            age_label = f"{age_start}-{age_end}"
        rows.append(
            {
                "IndicatorId": 79,
                "IndicatorName": "Age specific mortality rate m(x,n) - abridged",
                "IndicatorShortName": "Age-specific mortality rates by age groups and by sex",
                "Source": "World Population Prospects",
                "SourceYear": 2024,
                "Author": "United Nations Population Division",
                "LocationId": 364,
                "Location": "Iran (Islamic Republic of)",
                "Iso2": "IR",
                "Iso3": "IRN",
                "TimeId": 75,
                "Time": 2024,
                "VariantId": 4,
                "Variant": "Median",
                "SexId": 1,
                "Sex": "Male",
                "AgeId": 42,
                "AgeStart": age_start,
                "AgeEnd": age_end,
                "Age": age_label,
                "CategoryId": 0,
                "Category": "Not applicable",
                "EstimateTypeId": 1,
                "EstimateType": "Model-based Estimates",
                "EstimateMethodId": 2,
                "EstimateMethod": "Interpolation",
                "Value": rate,
            }
        )
    path = tmp_path / "wpp_iran.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Population loader tests
# ---------------------------------------------------------------------------


def test_load_iran_population_handles_open_age_group(synthetic_population):
    path_t, _ = synthetic_population
    df = load_iran_population(path_t)
    assert "Age" in df.columns
    assert 100 in df["Age"].to_list()
    assert "M" in df.columns and "F" in df.columns and "Total" in df.columns
    row_100 = df.filter(pl.col("Age") == 100)
    assert (
        int(row_100["M"].item())
        + int(row_100["F"].item())
        == int(row_100["Total"].item())
    )


# ---------------------------------------------------------------------------
# Cohort method tests
# ---------------------------------------------------------------------------


def _aligned_population(path: str | Path, sex: str) -> tuple[pl.Series, pl.Series]:
    """Load a population CSV and return aligned (pop_t, pop_t1) series."""
    df = load_iran_population(path)
    col = {"total": "Total", "male": "M", "female": "F"}[sex]
    return df[col].cast(pl.Float64), df[col].cast(pl.Float64)


def test_compute_cohort_mortality_recovers_known_rate(synthetic_population):
    path_t, path_t1 = synthetic_population
    pop_t = load_iran_population(path_t)["Total"].cast(pl.Float64)
    pop_t1 = load_iran_population(path_t1)["Total"].cast(pl.Float64)
    rates = compute_cohort_mortality(pop_t, pop_t1)
    valid = rates.drop_nulls()
    assert len(valid) > 0
    # 1% tolerance for integer rounding noise at small pop counts.
    np.testing.assert_allclose(valid.to_numpy(), 0.1, atol=1e-2)


def test_compute_cohort_mortality_handles_in_migration():
    # Cohort method pairs pop_t[x] with pop_t1[x+1]. In-migration for
    # the 1 -> 2 transition means pop_t1[2] > pop_t[1] => q < 0 => null.
    pop_t = pl.Series([1000.0, 1000.0, 1000.0, 50.0])
    pop_t1 = pl.Series([1000.0, 1000.0, 1100.0, 45.0])
    rates = compute_cohort_mortality(pop_t, pop_t1).to_list()
    assert rates[0] == 0.0
    assert math.isnan(rates[1])
    assert math.isnan(rates[3]) is False  # open interval
    np.testing.assert_allclose(rates[3], 0.1, atol=1e-9)


def test_aggregate_to_poland_buckets_uniform_population():
    rates = pl.Series([0.05] * 101, dtype=pl.Float64)
    pops = pl.Series([1.0] * 101, dtype=pl.Float64)
    ages = pl.Series(list(range(101)), dtype=pl.Int64)
    buckets = aggregate_to_poland_buckets(rates, pops, ages)
    assert set(buckets) == {label for label, _ in POLAND_AGE_BUCKETS}
    assert buckets["0"] == pytest.approx(0.05)
    assert buckets["1-4"] == pytest.approx(0.05)
    assert buckets["85-89"] == pytest.approx(0.05)
    assert buckets["90+"] == pytest.approx(0.05)


def test_aggregate_to_poland_buckets_population_weighted():
    ages = pl.Series([0, 1, 2, 3, 4, 5], dtype=pl.Int64)
    rates = pl.Series([0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=pl.Float64)
    pops = pl.Series([100.0, 100.0, 100.0, 100.0, 1.0, 100.0], dtype=pl.Float64)
    buckets = aggregate_to_poland_buckets(rates, pops, ages)
    # 1-4 bucket: ages 1,2,3,4 with pops 100,100,100,1. Weighted rate
    # = 0.5*1 / (100+100+100+1) = 0.5/301
    assert buckets["1-4"] == pytest.approx(0.5 / 301)


def test_aggregate_to_poland_buckets_skips_null_in_migration():
    ages = pl.Series([0, 1, 2, 3, 4, 5, 6], dtype=pl.Int64)
    rates = pl.Series(
        [0.0, None, None, None, None, 0.5, 0.0], dtype=pl.Float64
    )
    pops = pl.Series(
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=pl.Float64
    )
    buckets = aggregate_to_poland_buckets(rates, pops, ages)
    # All four ages in 1-4 are null -> bucket is null.
    assert buckets["1-4"] is None
    # 5-9 bucket: ages 5 (rate 0.5, pop 100) and 6 (rate 0, pop 100)
    # population-weighted = (0.5*100 + 0*100) / 200 = 0.25.
    assert buckets["5-9"] == pytest.approx(0.25)


def test_compute_crude_annual_rate_matches_known_input():
    # The cohort method treats the last element of the series as the
    # open age group (100+). With 10 elements, the cohort pairs:
    #   pop_t[0..7] with pop_t1[1..8]  (closed ages 0..7)
    #   pop_t[8]    with pop_t1[9]    (closed age 8 -> open interval)
    #   pop_t[9]    with pop_t1[9]    (open interval exit rate)
    # With pop_t1[i+1] = 950 for i=0..7, pop_t1[9] = 90:
    #   deaths = 8*50 + (1000-90) + (100-90) = 400 + 910 + 10 = 1320
    #   pop    = 9*1000 + 100 = 9100
    pop_t = pl.Series([1000.0] * 9 + [100.0])
    pop_t1 = pl.Series([1000.0] + [950.0] * 8 + [90.0])
    rate = compute_crude_annual_rate(pop_t, pop_t1)
    assert rate == pytest.approx(1320 / 9100)


# ---------------------------------------------------------------------------
# Builder + schema tests
# ---------------------------------------------------------------------------


def test_build_mortality_table_schema(synthetic_population):
    path_t, path_t1 = synthetic_population
    table = build_mortality_table(path_t, path_t1, sex="total")
    assert table["use_age_specific"] is True
    assert "crude_annual_rate" in table
    assert "age_specific" in table
    assert "source" in table
    assert table["source"]["method"] == "cohort_survival"
    assert table["source"]["sex"] == "total"
    expected_labels = {label for label, _ in POLAND_AGE_BUCKETS}
    assert set(table["age_specific"]) == expected_labels


def test_build_mortality_table_sex_split(synthetic_population):
    path_t, path_t1 = synthetic_population
    for sex in ("male", "female", "total"):
        table = build_mortality_table(path_t, path_t1, sex=sex)
        assert table["source"]["sex"] == sex
        for label, rate in table["age_specific"].items():
            assert rate == pytest.approx(0.1, abs=1e-2), label


def test_write_mortality_json_roundtrip(tmp_path, synthetic_population):
    path_t, path_t1 = synthetic_population
    table = build_mortality_table(path_t, path_t1, sex="total")
    out = tmp_path / "out.json"
    write_mortality_json(table, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["crude_annual_rate"] == table["crude_annual_rate"]
    assert loaded["age_specific"] == table["age_specific"]


def test_write_mortality_json_emits_null_for_nan(tmp_path):
    out = tmp_path / "out.json"
    write_mortality_json(
        {
            "use_age_specific": True,
            "crude_annual_rate": float("nan"),
            "age_specific": {"0": 0.001, "1-4": float("nan")},
            "source": {"country": "Iran", "method": "cohort_survival"},
        },
        out,
    )
    raw = out.read_text(encoding="utf-8")
    assert "null" in raw
    loaded = json.loads(raw)
    assert loaded["crude_annual_rate"] is None
    assert loaded["age_specific"]["1-4"] is None
    assert loaded["age_specific"]["0"] == 0.001


def test_compare_to_reference_with_real_poland_file(synthetic_population):
    path_t, path_t1 = synthetic_population
    iran = build_mortality_table(path_t, path_t1, sex="total")
    poland = PROJECT_ROOT / "data" / "mortality.json"
    if not poland.exists():
        pytest.skip("data/mortality.json not present")
    df = compare_to_reference(iran, poland)
    assert df.columns == [
        "age_bucket",
        "iran_per_1000",
        "reference_per_1000",
        "ratio_iran_over_ref",
    ]
    assert len(df) == len(POLAND_AGE_BUCKETS)
    assert df["age_bucket"].to_list()[0] == "0"
    assert df["age_bucket"].to_list()[-1] == "90+"


# ---------------------------------------------------------------------------
# Real-data integration test (skipped if data not present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not IRAN_DIR.exists(), reason="Iran data not available")
def test_build_mortality_table_real_iran_data():
    csv_t = IRAN_DIR / "Iran_2023.csv"
    csv_t1 = IRAN_DIR / "Iran_2024.csv"
    if not (csv_t.exists() and csv_t1.exists()):
        pytest.skip("Iran_<year>.csv files not present")
    table = build_mortality_table(csv_t, csv_t1, sex="total")
    crude = table["crude_annual_rate"]
    assert 0.002 < crude < 0.020
    rates_list = [
        r for r in table["age_specific"].values() if r is not None and math.isfinite(r)
    ]
    assert min(rates_list) < max(rates_list)


# ---------------------------------------------------------------------------
# UN WPP tests
# ---------------------------------------------------------------------------


def test_load_wpp_mortality_filters_correctly(synthetic_wpp_csv):
    df = load_wpp_mortality(synthetic_wpp_csv, year=2024, sex="Male")
    assert len(df) == len(_SYNTHETIC_RATES)
    rows_by_age = {row["AgeStart"]: row for row in df.iter_rows(named=True)}
    for age_start, row in rows_by_age.items():
        if age_start in WPP_OPEN_AGE_STARTS:
            assert row["poland_label"] == "90+"
        else:
            assert row["poland_label"] == WPP_AGE_START_TO_POLAND[age_start]


def test_load_wpp_mortality_rejects_wrong_year(synthetic_wpp_csv):
    with pytest.raises(ValueError, match="No rows"):
        load_wpp_mortality(synthetic_wpp_csv, year=1900, sex="Male")


def test_load_wpp_mortality_rejects_wrong_sex(synthetic_wpp_csv):
    with pytest.raises(ValueError, match="No rows"):
        load_wpp_mortality(synthetic_wpp_csv, year=2024, sex="Female")


def test_build_mortality_from_wpp_directly(synthetic_wpp_csv):
    table = build_mortality_from_wpp(synthetic_wpp_csv, year=2024, sex="Male")
    assert table["source"]["method"] == "un_wpp_abridged"
    assert table["source"]["sex"] == "Male"
    assert set(table["age_specific"]) == {label for label, _ in POLAND_AGE_BUCKETS}
    for label, rate in table["age_specific"].items():
        assert rate is not None
        assert math.isfinite(rate), label
    for age_start, label in WPP_AGE_START_TO_POLAND.items():
        np.testing.assert_allclose(
            table["age_specific"][label], _SYNTHETIC_RATES[age_start], rtol=1e-9
        )


def test_build_mortality_from_wpp_combines_90_plus(synthetic_wpp_csv):
    table = build_mortality_from_wpp(synthetic_wpp_csv, year=2024, sex="Male")
    rate_90 = _SYNTHETIC_RATES[90]
    rate_95 = _SYNTHETIC_RATES[95]
    rate_100 = _SYNTHETIC_RATES[100]
    assert min(rate_90, rate_95, rate_100) <= table["age_specific"]["90+"] <= max(
        rate_90, rate_95, rate_100
    )


def test_build_mortality_from_wpp_with_pop_weighting(
    synthetic_population, synthetic_wpp_csv
):
    csv_t, _ = synthetic_population
    table = build_mortality_from_wpp(
        synthetic_wpp_csv, year=2024, sex="Male", pop_csv=csv_t
    )
    assert math.isfinite(table["age_specific"]["90+"])
    assert math.isfinite(table["crude_annual_rate"])


def test_merge_cohort_with_wpp_fills_null_buckets(
    synthetic_population, synthetic_wpp_csv
):
    csv_t, csv_t1 = synthetic_population
    cohort = build_mortality_table(csv_t, csv_t1, sex="male")
    cohort["age_specific"]["0"] = float("nan")
    cohort["age_specific"]["1-4"] = None
    merged = merge_cohort_with_wpp(cohort, synthetic_wpp_csv, year=2024, sex="Male")
    assert math.isfinite(merged["age_specific"]["0"])
    assert math.isfinite(merged["age_specific"]["1-4"])
    assert "90+" not in merged["source"]["wpp_filled"]
    assert "0" in merged["source"]["wpp_filled"]
    assert "1-4" in merged["source"]["wpp_filled"]


def test_merge_cohort_with_wpp_preserves_existing_values(
    synthetic_population, synthetic_wpp_csv
):
    csv_t, csv_t1 = synthetic_population
    cohort = build_mortality_table(csv_t, csv_t1, sex="male")
    merged = merge_cohort_with_wpp(cohort, synthetic_wpp_csv, year=2024, sex="Male")
    assert merged["source"]["wpp_filled"] == []
    for label, rate in cohort["age_specific"].items():
        assert merged["age_specific"][label] == rate


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def _build_clean_table() -> dict:
    """A minimal but fully valid mortality table for validation tests."""
    return {
        "use_age_specific": True,
        "crude_annual_rate": 0.005,
        "age_specific": {
            label: 0.001 + 0.001 * i
            for i, (label, _) in enumerate(POLAND_AGE_BUCKETS)
        },
        "source": {"method": "test"},
    }


def test_validate_clean_table_returns_no_warnings():
    assert validate_mortality_table(_build_clean_table()) == []


def test_validate_strict_clean_table_does_not_raise():
    validate_mortality_table(_build_clean_table(), strict=True)


def test_validate_warns_on_missing_bucket():
    table = _build_clean_table()
    del table["age_specific"]["40-44"]
    warnings = validate_mortality_table(table)
    assert any("Missing Poland buckets" in w and "40-44" in w for w in warnings)


def test_validate_strict_raises_on_missing_bucket():
    table = _build_clean_table()
    del table["age_specific"]["40-44"]
    with pytest.raises(ValueError, match="Missing Poland buckets"):
        validate_mortality_table(table, strict=True)


def test_validate_warns_on_extra_bucket():
    table = _build_clean_table()
    table["age_specific"]["200+"] = 0.99
    warnings = validate_mortality_table(table)
    assert any("Unexpected buckets" in w and "200+" in w for w in warnings)


def test_validate_warns_on_rate_zero():
    table = _build_clean_table()
    table["age_specific"]["0"] = 0.0
    warnings = validate_mortality_table(table)
    assert any("outside (0, 1]" in w and "'0'" in w for w in warnings)


def test_validate_warns_on_rate_above_one():
    table = _build_clean_table()
    table["age_specific"]["90+"] = 1.5
    warnings = validate_mortality_table(table)
    assert any("outside (0, 1]" in w and "'90+'" in w for w in warnings)


def test_validate_strict_raises_on_rate_out_of_range():
    table = _build_clean_table()
    table["age_specific"]["0"] = -0.1
    with pytest.raises(ValueError, match="outside"):
        validate_mortality_table(table, strict=True)


def test_validate_warns_on_non_finite_rate():
    table = _build_clean_table()
    table["age_specific"]["0"] = math.nan
    warnings = validate_mortality_table(table)
    assert any("non-finite" in w for w in warnings)


def test_validate_warns_on_crude_outside_iran_range():
    table = _build_clean_table()
    table["crude_annual_rate"] = 0.5
    warnings = validate_mortality_table(table)
    assert any("Crude rate" in w and "outside" in w for w in warnings)


def test_validate_strict_raises_on_crude_outside_iran_range():
    table = _build_clean_table()
    table["crude_annual_rate"] = 0.0001
    with pytest.raises(ValueError, match="Crude rate"):
        validate_mortality_table(table, strict=True)


def test_validate_accepts_none_buckets():
    table = _build_clean_table()
    table["age_specific"]["0"] = None
    assert validate_mortality_table(table) == []


def test_validate_real_iran_output_passes():
    """End-to-end: the real reconstructed Male WPP CSV + real Iran
    2024 population should pass validation without warnings."""
    wpp_csv = PROJECT_ROOT / "data" / "raw" / "population-un-data-portal-iran.csv"
    pop_csv = PROJECT_ROOT / "data" / "raw" / "Iran_2024.csv"
    if not (wpp_csv.exists() and pop_csv.exists()):
        pytest.skip("Real Iran WPP / population data not present")
    table = build_mortality_from_wpp(
        wpp_csv, year=2024, sex="Male", pop_csv=pop_csv
    )
    warnings = validate_mortality_table(table)
    assert warnings == [], f"Real WPP output should be clean; got: {warnings}"


def test_validate_real_poland_file_passes():
    poland = PROJECT_ROOT / "data" / "mortality.json"
    if not poland.exists():
        pytest.skip("data/mortality.json not present")
    table = json.loads(poland.read_text(encoding="utf-8"))
    assert validate_mortality_table(table) == []
