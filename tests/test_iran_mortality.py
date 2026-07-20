"""Tests for the Iran mortality calculator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    # age 0 in pop_t1 represents new births, unrelated to pop_t
    pop_t1[0] = max(int(round(0.9 * pop_t[0])), 1)
    for x in range(1, 100):
        pop_t1[x] = max(int(round(0.9 * pop_t[x - 1])), 1)
    # open age group: people aged 99 at t who survived become 100+ at t+1
    pop_t1[100] = max(int(round(0.9 * pop_t[99])), 1)

    def _frame(pop: dict[int, int]) -> pd.DataFrame:
        rows = []
        for a, n in pop.items():
            age_str = "100+" if a == 100 else str(a)
            half = n // 2
            rows.append({"Age": age_str, "M": half, "F": n - half})
        return pd.DataFrame(rows)

    path_t = tmp_path / "Iran_2023.csv"
    path_t1 = tmp_path / "Iran_2024.csv"
    _frame(pop_t).to_csv(path_t, index=False)
    _frame(pop_t1).to_csv(path_t1, index=False)
    return path_t, path_t1


def test_load_iran_population_handles_open_age_group(synthetic_population):
    path_t, _ = synthetic_population
    df = load_iran_population(path_t)
    assert df.index.name == "Age"
    assert 100 in df.index
    assert "M" in df.columns and "F" in df.columns and "Total" in df.columns
    assert int(df.loc[100, "M"]) + int(df.loc[100, "F"]) == int(df.loc[100, "Total"])


def test_compute_cohort_mortality_recovers_known_rate(synthetic_population):
    path_t, path_t1 = synthetic_population
    pop_t = load_iran_population(path_t)["Total"].astype(float)
    pop_t1 = load_iran_population(path_t1)["Total"].astype(float)
    rates = compute_cohort_mortality(pop_t, pop_t1)
    valid = rates.dropna()
    assert len(valid) > 0
    # Rounding of the integer cohort counts introduces ~1e-4 relative
    # error at small populations, so use a 1% tolerance. Cast via
    # np.asarray so Pylance can pick the right assert_allclose overload.
    np.testing.assert_allclose(np.asarray(valid.values), 0.1, atol=1e-2)


def test_compute_cohort_mortality_handles_in_migration():
    # Cohort method pairs pop_t[x] with pop_t1[x+1]. So simulate
    # in-migration for the age-1 -> age-2 transition by making
    # pop_t1[2] > pop_t[1].
    pop_t = pd.Series({0: 1000, 1: 1000, 2: 1000, 100: 50}, dtype=float)
    pop_t1 = pd.Series({0: 1000, 1: 1000, 2: 1100, 100: 45}, dtype=float)
    rates = compute_cohort_mortality(pop_t, pop_t1)
    # In-migration for age 1 -> 2 means q < 0; should be NaN
    assert np.isnan(rates.loc[1])
    # Age 0 -> 1: pop_t1[1] = 1000, pop_t[0] = 1000, q = 0
    assert rates.loc[0] == 0.0
    # 50 -> 45 in open group: q = 0.1
    np.testing.assert_allclose(rates.loc[100], 0.1)


def test_aggregate_to_poland_buckets_uniform_population():
    rates = pd.Series({a: 0.05 for a in range(101)}, dtype=float)
    pops = pd.Series({a: 1.0 for a in range(101)}, dtype=float)
    buckets = aggregate_to_poland_buckets(rates, pops)
    assert set(buckets) == {label for label, _ in POLAND_AGE_BUCKETS}
    assert buckets["0"] == pytest.approx(0.05)
    assert buckets["1-4"] == pytest.approx(0.05)
    assert buckets["85-89"] == pytest.approx(0.05)
    assert buckets["90+"] == pytest.approx(0.05)


def test_aggregate_to_poland_buckets_population_weighted():
    rates = pd.Series(
        {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.5, 5: 0.0}, dtype=float
    )
    pops = pd.Series({0: 100, 1: 100, 2: 100, 3: 100, 4: 1, 5: 100}, dtype=float)
    buckets = aggregate_to_poland_buckets(rates, pops)
    # 1-4 bucket: ages 1,2,3,4 with pops 100,100,100,1. Weighted rate
    # = 0.5*1 / (100+100+100+1) = 0.5/301
    assert buckets["1-4"] == pytest.approx(0.5 / 301)


def test_aggregate_to_poland_buckets_skips_nan_in_migration():
    # All NaN in the 1-4 range, valid rate at 5+. Bucket 1-4 should be
    # NaN; bucket 5-9 should pick up age 5.
    rates = pd.Series(
        {0: 0.0, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: 0.5, 6: 0.0}, dtype=float
    )
    pops = pd.Series(
        {0: 100, 1: 100, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100}, dtype=float
    )
    buckets = aggregate_to_poland_buckets(rates, pops)
    assert np.isnan(buckets["1-4"])
    assert buckets["5-9"] == pytest.approx(0.5 * 100 / 200)


def test_compute_crude_annual_rate_matches_known_input():
    pop_t = pd.Series({0: 1000, 1: 1000, 2: 1000, 100: 50}, dtype=float)
    pop_t1 = pd.Series({0: 1000, 1: 900, 2: 1000, 100: 45}, dtype=float)
    rate = compute_crude_annual_rate(pop_t, pop_t1)
    # deaths: (1000-900) + 0 + (50-45) = 105; pop: 3050
    assert rate == pytest.approx(105 / 3050)


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
        # 10% annual drop -> bucket rate ~0.1 everywhere. Loose tolerance
        # because rounding noise compounds at small pop counts.
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
    assert list(df.columns) == [
        "age_bucket",
        "iran_per_1000",
        "reference_per_1000",
        "ratio_iran_over_ref",
    ]
    assert len(df) == len(POLAND_AGE_BUCKETS)
    assert df["age_bucket"].iloc[0] == "0"
    assert df["age_bucket"].iloc[-1] == "90+"


@pytest.mark.skipif(not IRAN_DIR.exists(), reason="Iran data not available")
def test_build_mortality_table_real_iran_data():
    csv_t = IRAN_DIR / "Iran_2023.csv"
    csv_t1 = IRAN_DIR / "Iran_2024.csv"
    if not (csv_t.exists() and csv_t1.exists()):
        pytest.skip("Iran_<year>.csv files not present")
    table = build_mortality_table(csv_t, csv_t1, sex="total")
    crude = table["crude_annual_rate"]
    # Crude rate for Iran should be in a sane range (0.004 - 0.012)
    assert 0.002 < crude < 0.020
    rates = np.array(list(table["age_specific"].values()), dtype=float)
    rates = rates[~np.isnan(rates)]
    # Mortality must be non-decreasing-ish in the young adult range
    assert rates[1] < rates[-1]


# ---------------------------------------------------------------------------
# UN WPP tests
# ---------------------------------------------------------------------------

WPP_HEADER = (
    "IndicatorId,IndicatorName,IndicatorShortName,Source,SourceYear,Author,"
    "LocationId,Location,Iso2,Iso3,TimeId,Time,VariantId,Variant,SexId,Sex,"
    "AgeId,AgeStart,AgeEnd,Age,CategoryId,Category,EstimateTypeId,"
    "EstimateType,EstimateMethodId,EstimateMethod,Value"
)

# Rates that increase with age (synthetic but realistic for an adult male)
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
    import csv

    rows = []
    for age_start, rate in _SYNTHETIC_RATES.items():
        if age_start == 0:
            age_end = 1
            age_label = "0"
        elif age_start == 1:
            age_end = 5
            age_label = "1-4"
        elif age_start == 100:
            age_end = 100
            age_label = "100+"
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


def test_load_wpp_mortality_filters_correctly(synthetic_wpp_csv):
    df = load_wpp_mortality(synthetic_wpp_csv, year=2024, sex="Male")
    assert len(df) == len(_SYNTHETIC_RATES)
    # All 5-year buckets should map to a Poland label
    for age_start, row in df.iterrows():
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
    # All 20 Poland buckets should be present and finite
    assert set(table["age_specific"]) == {label for label, _ in POLAND_AGE_BUCKETS}
    for label, rate in table["age_specific"].items():
        assert rate is not None
        assert np.isfinite(rate), label
    # Direct 5-year buckets should match the synthetic data exactly
    for age_start, label in WPP_AGE_START_TO_POLAND.items():
        np.testing.assert_allclose(
            table["age_specific"][label], _SYNTHETIC_RATES[age_start], rtol=1e-9
        )


def test_build_mortality_from_wpp_combines_90_plus(synthetic_wpp_csv):
    table = build_mortality_from_wpp(synthetic_wpp_csv, year=2024, sex="Male")
    rate_90 = _SYNTHETIC_RATES[90]
    rate_95 = _SYNTHETIC_RATES[95]
    rate_100 = _SYNTHETIC_RATES[100]
    # Without population weighting the fallback uses 95-99
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
    # 90+ should still be in the valid range
    assert np.isfinite(table["age_specific"]["90+"])
    # Crude rate should now be finite too
    assert np.isfinite(table["crude_annual_rate"])


def test_merge_cohort_with_wpp_fills_null_buckets(
    synthetic_population, synthetic_wpp_csv
):
    csv_t, csv_t1 = synthetic_population
    cohort = build_mortality_table(csv_t, csv_t1, sex="male")
    # Replace the 0 and 90+ buckets with NaN to simulate unreliable data
    cohort["age_specific"]["0"] = float("nan")
    cohort["age_specific"]["1-4"] = None
    merged = merge_cohort_with_wpp(cohort, synthetic_wpp_csv, year=2024, sex="Male")
    assert np.isfinite(merged["age_specific"]["0"])
    assert np.isfinite(merged["age_specific"]["1-4"])
    # 90+ was finite in the synthetic cohort so it should not be in the filled list
    assert "90+" not in merged["source"]["wpp_filled"]
    # The two we nulled should be reported as filled
    assert "0" in merged["source"]["wpp_filled"]
    assert "1-4" in merged["source"]["wpp_filled"]


def test_merge_cohort_with_wpp_preserves_existing_values(
    synthetic_population, synthetic_wpp_csv
):
    csv_t, csv_t1 = synthetic_population
    cohort = build_mortality_table(csv_t, csv_t1, sex="male")
    # All buckets are finite in the synthetic cohort; nothing should be filled
    merged = merge_cohort_with_wpp(cohort, synthetic_wpp_csv, year=2024, sex="Male")
    assert merged["source"]["wpp_filled"] == []
    # Cohort values should be preserved exactly
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
    table = _build_clean_table()
    assert validate_mortality_table(table) == []


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
    import math

    table = _build_clean_table()
    table["age_specific"]["0"] = math.nan
    warnings = validate_mortality_table(table)
    assert any("non-finite" in w for w in warnings)


def test_validate_warns_on_crud_outside_iran_range():
    table = _build_clean_table()
    table["crude_annual_rate"] = 0.5
    warnings = validate_mortality_table(table)
    assert any("Crude rate" in w and "outside" in w for w in warnings)


def test_validate_strict_raises_on_crud_outside_iran_range():
    table = _build_clean_table()
    table["crude_annual_rate"] = 0.0001
    with pytest.raises(ValueError, match="Crude rate"):
        validate_mortality_table(table, strict=True)


def test_validate_accepts_none_buckets():
    table = _build_clean_table()
    table["age_specific"]["0"] = None
    # None is allowed; it represents "unreliable" rather than an error.
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
