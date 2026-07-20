"""Iran age-specific mortality calculator for the hemophilia Markov model.

This module produces an age-specific mortality table in the same shape as
``data/mortality.json`` (Poland) so it can be dropped into the model
without code changes. Two methods are implemented:

1. **Cohort survival method** (Preston, Heuveline & Guillot, 2001,
   §3.4). Given two population snapshots ``P(x, t)`` and
   ``P(x+1, t+1)`` at single years of age, the probability of dying
   in the interval is ::

        q(x) = 1 - P(x + 1, t + 1) / P(x, t)

   For the open age group ``x = 100``, the same formula compares the
   100+ populations at the two time points, giving the one-year
   "exit" rate of the open interval. Negative values arise when
   ``P(x+1, t+1) > P(x, t)`` (population grew), which happens for
   any cohort that experienced net in-migration or a birth pulse.
   We mark those ages as ``NaN`` (rendered as ``null`` in JSON)
   because we genuinely cannot disentangle growth from zero deaths.
   This makes the cohort method unsuitable for the youngest ages in
   expanding populations such as Iran 2014-2024; it is the
   recommended method for older ages and for any closed population.

2. **UN WPP 2024 abridged life table** (United Nations DESA, 2024).
   The WPP publishes age-specific mortality rates ``m(x, n)`` in
   five-year buckets (with a one-year bucket for age 0 and an open
   bucket for 100+). These are model-based estimates that absorb
   vital registration, sibling histories, and indirect estimation
   techniques; they are the gold standard for countries like Iran
   where direct cohort survival is unreliable due to migration.

   Because hemophilia is X-linked recessive and severe disease is
   essentially male-only, the Male mortality curve is the right
   input for the modeled cohort. The calculator reads the WPP
   export, optionally population-weights the open 90+ buckets
   (``90-94``, ``95-99``, ``100+``) into the single ``"90+"``
   bucket of the Poland schema, and emits the table directly.

A third mode (``merge_cohort_with_wpp``) keeps the cohort-derived
value where it is reliable and fills the ``NaN`` buckets from WPP,
so a single output combines the best of both.

Output units
------------
All rates are deaths per person-year. So ``0.00369`` is 3.69 per
1,000 person-years, matching the units of the existing Poland table.

Validation
----------
The :func:`validate_mortality_table` function checks the structure
(rate range, bucket coverage) and round-trips through the Pydantic
``MortalityFile`` schema. It returns a list of warnings rather than
raising, so the CLI can print a clean summary.

How to cite
-----------
If this module is used in published work, please cite both:

* United Nations, Department of Economic and Social Affairs,
  Population Division (2024). *World Population Prospects 2024*.
  Online edition. https://population.un.org/wpp/
* Preston, S. H., Heuveline, P., & Guillot, M. (2001).
  *Demography: Measuring and Modeling Population Processes*.
  Blackwell, Chapter 3 (cohort-component method).

And for the input dataset:

* Statistical Centre of Iran (SCI), national population estimates
  (the ``Iran_<year>.csv`` files in ``data/raw/``).

Limitations and assumptions
---------------------------
* The cohort method assumes a **closed population** (no net
  migration). Iran's net emigration of young adults over 2014-2024
  means young-adult rates would be over-estimated if not for the
  early termination of the method when ``P(x+1, t+1) > P(x, t)``.
* The WPP rates are model-based estimates, not direct observations.
  They smooth over subnational variation and use vital registration
  data of varying quality; the calculator treats them as point
  estimates with no propagated uncertainty.
* The 90+ combination uses population weights from
  ``Iran_<year>.csv``. If the population file is not provided the
  calculator falls back to the 95-99 rate as a mid-range proxy.
* The bucket aggregation treats the WPP ``m(x, n)`` (central death
  rate) as interchangeable with the per-person-year rate ``q(x)``
  used by the Markov model. For small rates the difference is
  negligible (< 1% for ``m < 0.05``), which covers every bucket
  except 90+; there the 90+ combination is dominated by the 95-99
  and 100+ values which exceed 0.30 and the approximation over-
  estimates the per-year rate by 5-10 %. For the hemophilia model
  the 90+ bucket is rarely reached so the impact is small.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError

from persistence.schemas.mortality import MortalityFile

__version__ = "1.0.0"
__all__ = [
    # Constants
    "POLAND_AGE_BUCKETS",
    "WPP_AGE_START_TO_POLAND",
    "WPP_OPEN_AGE_STARTS",
    # Public API
    "load_iran_population",
    "load_wpp_mortality",
    "build_mortality_table",
    "build_mortality_from_wpp",
    "merge_cohort_with_wpp",
    "write_mortality_json",
    "validate_mortality_table",
    "compare_to_reference",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sex literals accepted by the cohort method.
Sex = Literal["total", "male", "female"]

#: Sex labels as they appear in the WPP "Sex" column.
WPPSex = Literal["Male", "Female", "Both sexes"]

#: Age-bucket labels and the integer ages they cover. The first entry
#: is the single-year bucket for age 0; the last is the open 90+ bucket
#: covering ages 90-100 (where 100 is the open "100+" group from the
#: Iran population files).
POLAND_AGE_BUCKETS: list[tuple[str, range]] = [
    ("0", range(0, 1)),
    ("1-4", range(1, 5)),
    ("5-9", range(5, 10)),
    ("10-14", range(10, 15)),
    ("15-19", range(15, 20)),
    ("20-24", range(20, 25)),
    ("25-29", range(25, 30)),
    ("30-34", range(30, 35)),
    ("35-39", range(35, 40)),
    ("40-44", range(40, 45)),
    ("45-49", range(45, 50)),
    ("50-54", range(50, 55)),
    ("55-59", range(55, 60)),
    ("60-64", range(60, 65)),
    ("65-69", range(65, 70)),
    ("70-74", range(70, 75)),
    ("75-79", range(75, 80)),
    ("80-84", range(80, 85)),
    ("85-89", range(85, 90)),
    ("90+", range(90, 101)),
]

#: Mapping from WPP ``AgeStart`` (lower bound of each 5-year bucket,
#: with 0 for the 0-1 single-year bucket) to the Poland schema label.
#: The 90, 95, and 100 rows are intentionally absent - they collapse
#: into the ``"90+"`` bucket via :data:`WPP_OPEN_AGE_STARTS`.
WPP_AGE_START_TO_POLAND: dict[int, str] = {
    0: "0",
    1: "1-4",
    5: "5-9",
    10: "10-14",
    15: "15-19",
    20: "20-24",
    25: "25-29",
    30: "30-34",
    35: "35-39",
    40: "40-44",
    45: "45-49",
    50: "50-54",
    55: "55-59",
    60: "60-64",
    65: "65-69",
    70: "70-74",
    75: "75-79",
    80: "80-84",
    85: "85-89",
}

#: The WPP ``AgeStart`` values that share the Poland ``"90+"`` bucket.
WPP_OPEN_AGE_STARTS: tuple[int, ...] = (90, 95, 100)

#: Lower-bound on a reasonable Iran crude death rate (deaths / person-year).
#: Used by :func:`validate_mortality_table` to flag suspiciously low tables.
_MIN_REASONABLE_CRUDE = 0.002

#: Upper-bound on a reasonable Iran crude death rate (deaths / person-year).
#: Iran's all-time low is around 0.004 (1950) and its all-time high is
#: around 0.025 (1980s war years); we use 0.05 as a generous ceiling.
_MAX_REASONABLE_CRUDE = 0.05


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_iran_population(path: str | Path) -> pd.DataFrame:
    """Load an ``Iran_<year>.csv`` population-by-age file.

    The file must have columns ``Age, M, F``. ``Age`` may contain the
    string ``"100+"`` for the open age group; this is coerced to the
    integer ``100``. Rows with missing ``Age``, ``M``, or ``F`` are
    dropped.

    Parameters
    ----------
    path
        Path to ``Iran_<year>.csv``.

    Returns
    -------
    pandas.DataFrame
        Indexed by integer age (0..100). Columns ``M``, ``F``, and
        ``Total = M + F``.
    """
    df = pd.read_csv(path)
    if not {"Age", "M", "F"}.issubset(df.columns):
        raise ValueError(
            f"{path} is missing required columns; expected Age, M, F; "
            f"got {list(df.columns)}"
        )
    df["Age"] = df["Age"].replace("100+", "100")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["M"] = pd.to_numeric(df["M"], errors="coerce")
    df["F"] = pd.to_numeric(df["F"], errors="coerce")
    df = df.dropna(subset=["Age", "M", "F"])
    df["Age"] = df["Age"].astype(int)
    df["M"] = df["M"].astype(int)
    df["F"] = df["F"].astype(int)
    df["Total"] = df["M"] + df["F"]
    return df.set_index("Age")[["M", "F", "Total"]].sort_index()


def load_wpp_mortality(
    csv_path: str | Path,
    year: int,
    sex: WPPSex = "Male",
    indicator_id: int = 79,
) -> pd.DataFrame:
    """Load a UN Data Portal abridged-life-table export for Iran.

    The file is expected to be the CSV download from
    https://population.un.org/dataportal/ with the columns
    ``IndicatorId, Time, Variant, Sex, AgeStart, Age, EstimateType,
    Value``. Only ``IndicatorId == 79`` (abridged), ``Variant ==
    "Median"``, and ``EstimateType == "Model-based Estimates"`` are
    retained so the result contains only observed / interpolated
    estimates (not projections, which would include 2025+ and
    onwards).

    Parameters
    ----------
    csv_path
        Path to the UN Data Portal export.
    year
        Calendar year of the estimate to read (e.g. ``2024``).
    sex
        ``"Male"``, ``"Female"``, or ``"Both sexes"``.
    indicator_id
        ``79`` for the abridged (5-year) life table, ``80`` for the
        complete (single-year) life table. Abridged is what matches
        the Poland schema.

    Returns
    -------
    pandas.DataFrame
        Indexed by integer ``AgeStart`` (0, 1, 5, ..., 100). Columns
        ``age_label`` (UN's human-readable bucket), ``value``
        (mortality rate, deaths per person-year), and ``poland_label``
        (the matching Poland bucket; rows for 90, 95, 100 all map
        to ``"90+"``).
    """
    df = pd.read_csv(csv_path)
    required = {"IndicatorId", "Time", "Variant", "Sex", "AgeStart", "Age", "EstimateType", "Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} doesn't look like a UN Data Portal export: "
            f"missing columns {sorted(missing)}"
        )
    filtered = df[
        (df["IndicatorId"] == indicator_id)
        & (df["Time"] == year)
        & (df["Sex"] == sex)
        & (df["Variant"] == "Median")
        & (df["EstimateType"] == "Model-based Estimates")
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows in {csv_path} match IndicatorId={indicator_id}, "
            f"Time={year}, Sex={sex!r}, Variant=Median, "
            f"EstimateType=Model-based Estimates. Did you forget to "
            f"expand the Sex / Age filters on the UN Data Portal?"
        )
    # AgeStart is always integer in the WPP export; cast explicitly
    # so downstream code can do arithmetic on it.
    filtered["AgeStart"] = filtered["AgeStart"].astype(int)
    filtered["Value"] = filtered["Value"].astype(float)
    filtered["poland_label"] = filtered["AgeStart"].map(WPP_AGE_START_TO_POLAND)
    # The three open-age rows (90, 95, 100) collapse to "90+".
    filtered.loc[filtered["AgeStart"].isin(WPP_OPEN_AGE_STARTS), "poland_label"] = "90+"
    out = (
        filtered[["AgeStart", "Age", "Value", "poland_label"]]
        .rename(columns={"Age": "age_label", "Value": "value"})
        .set_index("AgeStart")
        .sort_index()
    )
    return out


# ---------------------------------------------------------------------------
# Cohort survival method
# ---------------------------------------------------------------------------


def compute_cohort_mortality(
    pop_t: pd.Series,
    pop_t1: pd.Series,
) -> pd.Series:
    """Single-year mortality rate via the cohort survival method.

    For each age ``x`` present in ``pop_t``::

        q(x) = 1 - P(x_next, t + 1) / P(x, t)

    where ``x_next = x + 1`` for ``x < 100`` and ``x_next = 100``
    for ``x = 100`` (the open age group, so the formula gives the
    one-year exit rate of the open interval).

    Parameters
    ----------
    pop_t
        Population at exact ages at time ``t`` (integer index, values
        in persons). Should be a single-sex series (``M``, ``F``, or
        ``Total``).
    pop_t1
        Population at exact ages at time ``t + 1`` (same shape as
        ``pop_t``).

    Returns
    -------
    pandas.Series
        Indexed by ``x`` with one float value per age. ``NaN`` marks
        ages where the cohort grew (``P(x+1, t+1) > P(x, t)``) and
        the rate therefore cannot be estimated.
    """
    rates: dict[int, float] = {}
    for x in pop_t.index:
        next_x = x if x == 100 else x + 1
        if next_x not in pop_t1.index:
            rates[int(x)] = math.nan
            continue
        p_t = _to_float(pop_t.loc[x])
        p_t1 = _to_float(pop_t1.loc[next_x])
        if p_t <= 0:
            rates[int(x)] = math.nan
            continue
        q = 1.0 - p_t1 / p_t
        if q < 0.0:
            # Population grew; we cannot disentangle births/in-migration
            # from zero deaths. Mark as unreliable (NaN, not 0 - a 0
            # rate would be a quantitative claim, not a missing value).
            rates[int(x)] = math.nan
            continue
        # Pure-Python clamp to [0, 1]; avoids np.clip's broader Scalar
        # type which Pylance flags against float.__new__.
        rates[int(x)] = min(max(q, 0.0), 1.0)
    return pd.Series(rates, name="mortality_rate").sort_index()


def aggregate_to_poland_buckets(
    rates: pd.Series,
    populations: pd.Series,
) -> dict[str, float]:
    """Aggregate single-year ``q(x)`` to the 5-year buckets.

    Each bucket's rate is the **population-weighted mean** of its
    constituent single-year rates, so the result reflects the rate
    actually experienced by the cohort in the bucket. Buckets with
    no valid (non-NaN) single-year rates are returned as ``NaN``.

    Parameters
    ----------
    rates
        Single-year mortality rates, indexed by integer age. ``NaN``
        entries are skipped (and the whole bucket is ``NaN`` if every
        age in the bucket is ``NaN``).
    populations
        Population by single year of age (same index as ``rates``)
        used as the aggregation weights.

    Returns
    -------
    dict
        Mapping from Poland bucket label (``"0"``, ``"1-4"``, ...,
        ``"90+"``) to a float mortality rate, or ``NaN`` for
        unreliable buckets.
    """
    aggregated: dict[str, float] = {}
    for label, age_range in POLAND_AGE_BUCKETS:
        ages = [a for a in age_range if a in rates.index]
        if not ages:
            aggregated[label] = math.nan
            continue
        bucket_rates = rates.loc[ages]
        bucket_pops = populations.reindex(ages).fillna(0.0)
        valid = ~bucket_rates.isna()
        if not valid.any() or bucket_pops[valid].sum() <= 0:
            aggregated[label] = math.nan
            continue
        # weighted.mean = sum(rate * pop) / sum(pop)
        weighted = (bucket_rates[valid] * bucket_pops[valid]).sum()
        denom = bucket_pops[valid].sum()
        aggregated[label] = _to_float(weighted / denom)
    return aggregated


def compute_crude_annual_rate(
    pop_t: pd.Series,
    pop_t1: pd.Series,
) -> float:
    """Crude annual death rate (deaths per person-year) from two snapshots.

    Estimated as
    ``sum_x max(P_t[x] - P_{t+1}[x_next], 0) / sum_x P_t[x]``.
    The numerator counts only positive cohort changes (population
    *loss*) and treats growth as zero deaths, so the estimate is a
    lower bound on the true crude rate when migration is positive.
    """
    deaths = 0.0
    for x in pop_t.index:
        next_x = x if x == 100 else x + 1
        if next_x not in pop_t1.index:
            continue
        deaths += max(
            _to_float(pop_t.loc[x]) - _to_float(pop_t1.loc[next_x]),
            0.0,
        )
    total_pop = _to_float(pop_t.sum())
    if total_pop <= 0:
        return math.nan
    return deaths / total_pop


def build_mortality_table(
    iran_csv_t: str | Path,
    iran_csv_t1: str | Path,
    sex: Sex = "male",
) -> dict[str, Any]:
    """Build an age-specific mortality table from two Iran population snapshots.

    The returned dict matches the ``MortalityFile`` schema (see
    ``persistence/schemas/mortality.py``) with an extra ``source``
    block recording provenance.

    Parameters
    ----------
    iran_csv_t, iran_csv_t1
        Paths to two ``Iran_<year>.csv`` files. The years need not be
        literally consecutive, but the cohort method is sharpest when
        ``t1 - t == 1``.
    sex
        ``"total"`` (default for the model), ``"male"``, or
        ``"female"``.

    Returns
    -------
    dict
        ``{"use_age_specific": True, "crude_annual_rate": float|None,
        "age_specific": {label: float|None}, "source": {...}}``.
    """
    pop_t = load_iran_population(iran_csv_t)
    pop_t1 = load_iran_population(iran_csv_t1)

    if sex == "total":
        series_t, series_t1 = pop_t["Total"], pop_t1["Total"]
    elif sex == "male":
        series_t, series_t1 = pop_t["M"], pop_t1["M"]
    elif sex == "female":
        series_t, series_t1 = pop_t["F"], pop_t1["F"]
    else:
        raise ValueError(f"Unknown sex: {sex!r}")

    common = series_t.index.intersection(series_t1.index)
    series_t = series_t.loc[common].astype(float)
    series_t1 = series_t1.loc[common].astype(float)

    rates = compute_cohort_mortality(series_t, series_t1)
    aggregated = aggregate_to_poland_buckets(rates, series_t)
    crude = compute_crude_annual_rate(series_t, series_t1)

    return {
        "use_age_specific": True,
        "crude_annual_rate": crude,
        "age_specific": aggregated,
        "source": {
            "country": "Iran",
            "method": "cohort_survival",
            "pop_t": str(iran_csv_t),
            "pop_t1": str(iran_csv_t1),
            "sex": sex,
            "calculator_version": __version__,
        },
    }


# ---------------------------------------------------------------------------
# UN WPP method
# ---------------------------------------------------------------------------


def _combine_90_plus(
    wpp: pd.DataFrame,
    pop_csv: str | Path | None,
    sex: WPPSex = "Male",
) -> float | None:
    """Combine the 90-94, 95-99 and 100+ WPP buckets into a single
    population-weighted rate for the ``"90+"`` Poland bucket.

    Without a population file we fall back to the 95-99 rate (a
    mid-range value). With a population file we use the male
    (or female / total, depending on ``sex``) population in each
    5-year age range as the weight.
    """
    open_rows = wpp.loc[wpp.index.isin(WPP_OPEN_AGE_STARTS)]
    if open_rows.empty:
        return None
    if pop_csv is None:
        # Fallback: mid-range proxy. 95-99 is preferred; otherwise any.
        if 95 in open_rows.index:
            return _to_float(open_rows.loc[95, "value"])
        return _to_float(open_rows["value"].iloc[0])

    pop = load_iran_population(pop_csv)
    col = {"Male": "M", "Female": "F"}.get(sex, "Total")
    if col not in pop.columns:
        col = "Total"
    # Population at the 100+ open interval is reported as a single
    # age 100 row in the Iran_<year>.csv; summing 90:94 and 95:99
    # inclusive gives the matching closed 5-year buckets.
    weights: dict[int, int] = {
        90: int(_to_float(pop.loc[90:94, col].sum())),
        95: int(_to_float(pop.loc[95:99, col].sum())),
        100: int(_to_float(pop.loc[100, col])) if 100 in pop.index else 0,
    }
    total = sum(weights.values())
    if total <= 0:
        return _to_float(open_rows["value"].iloc[0])
    weighted = sum(
        weights[age_start] * _to_float(open_rows.loc[age_start, "value"])
        for age_start in WPP_OPEN_AGE_STARTS
        if age_start in open_rows.index
    )
    return float(weighted) / float(total)


def _crude_annual_rate_from_wpp(
    wpp: pd.DataFrame,
    pop_csv: str | Path | None,
    sex: WPPSex,
) -> float | None:
    """Crude annual death rate from the WPP rates + (optional) population.

    Iterating ``wpp.index`` directly (rather than ``iterrows()``)
    keeps the age typed as int - ``iterrows`` types the index as
    ``Hashable`` which Pylance rejects in ``+ 4`` expressions.
    """
    if pop_csv is None:
        return None
    pop = load_iran_population(pop_csv)
    col = {"Male": "M", "Female": "F"}.get(sex, "Total")
    if col not in pop.columns:
        col = "Total"
    deaths = 0.0
    total_pop = 0.0
    for age_start in wpp.index:
        rate = _to_float(wpp.loc[age_start, "value"])
        if age_start == 0:
            p = _to_float(pop.loc[0, col])
        elif age_start == 1:
            p = _to_float(pop.loc[1:4, col].sum())
        elif age_start in (90, 95):
            upper = age_start + 4
            p = _to_float(pop.loc[age_start:upper, col].sum())
        elif age_start == 100:
            p = _to_float(pop.loc[100, col]) if 100 in pop.index else 0.0
        else:
            upper = age_start + 4
            p = _to_float(pop.loc[age_start:upper, col].sum())
        total_pop += p
        deaths += rate * p
    if total_pop <= 0:
        return None
    return float(deaths) / float(total_pop)


def build_mortality_from_wpp(
    wpp_csv: str | Path,
    year: int,
    sex: WPPSex = "Male",
    pop_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build an age-specific mortality table directly from a UN WPP export.

    Parameters
    ----------
    wpp_csv
        Path to the UN Data Portal Iran export.
    year
        Calendar year of the estimate (e.g. ``2024``).
    sex
        ``"Male"`` (default, recommended for hemophilia models),
        ``"Female"``, or ``"Both sexes"``.
    pop_csv
        Optional path to an ``Iran_<year>.csv`` file used to
        population-weight the 90-94 / 95-99 / 100+ buckets into a
        single ``"90+"`` rate and to compute the crude rate.

    Returns
    -------
    dict
        ``MortalityFile``-compatible dict with a ``source`` block.
    """
    wpp = load_wpp_mortality(wpp_csv, year=year, sex=sex)
    aggregated: dict[str, float | None] = {}
    for label, _ in POLAND_AGE_BUCKETS:
        if label == "90+":
            aggregated[label] = _combine_90_plus(wpp, pop_csv, sex=sex)
            continue
        match = wpp[wpp["poland_label"] == label]
        if match.empty:
            aggregated[label] = None
        else:
            aggregated[label] = _to_float(match["value"].iloc[0])

    crude = _crude_annual_rate_from_wpp(wpp, pop_csv, sex=sex)

    return {
        "use_age_specific": True,
        "crude_annual_rate": crude,
        "age_specific": aggregated,
        "source": {
            "country": "Iran",
            "method": "un_wpp_abridged",
            "wpp_csv": str(wpp_csv),
            "year": int(year),
            "sex": sex,
            "pop_csv": str(pop_csv) if pop_csv else None,
            "calculator_version": __version__,
            "citation": (
                "United Nations DESA, Population Division (2024). "
                "World Population Prospects 2024."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Hybrid: cohort + WPP merge
# ---------------------------------------------------------------------------


def merge_cohort_with_wpp(
    cohort_table: dict[str, Any],
    wpp_csv: str | Path,
    year: int,
    sex: WPPSex = "Male",
    pop_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Fill any ``None``/``NaN``/missing buckets in ``cohort_table``
    with rates from the UN WPP export.

    The returned dict is a new table with the same schema; cohort-
    derived values that are present are kept as-is, and only the
    missing buckets are overwritten. The ``source`` block gains a
    ``wpp_filled`` list of bucket labels that were replaced so the
    hybrid provenance is auditable.
    """
    wpp_table = build_mortality_from_wpp(
        wpp_csv, year=year, sex=sex, pop_csv=pop_csv
    )
    merged: dict[str, Any] = {
        "use_age_specific": bool(cohort_table.get("use_age_specific", True)),
        "crude_annual_rate": cohort_table.get("crude_annual_rate"),
        "age_specific": dict(cohort_table.get("age_specific", {})),
        "source": dict(cohort_table.get("source", {})),
    }
    filled: list[str] = []
    for label, _ in POLAND_AGE_BUCKETS:
        current = merged["age_specific"].get(label)
        is_missing = current is None or (
            isinstance(current, float) and not math.isfinite(current)
        )
        wpp_value = wpp_table["age_specific"].get(label)
        if (
            is_missing
            and wpp_value is not None
            and math.isfinite(wpp_value)
        ):
            merged["age_specific"][label] = wpp_value
            filled.append(label)
    merged["source"]["wpp_filled"] = filled
    merged["source"]["wpp_csv"] = str(wpp_csv)
    merged["source"]["wpp_year"] = int(year)
    merged["source"]["calculator_version"] = __version__
    return merged


# ---------------------------------------------------------------------------
# I/O and validation
# ---------------------------------------------------------------------------


def write_mortality_json(table: dict[str, Any], output: str | Path) -> None:
    """Write the mortality table as JSON.

    ``NaN``/inf values in ``age_specific`` and ``crude_annual_rate``
    are emitted as JSON ``null`` so downstream parsers can detect the
    unreliable buckets and fill them in from another source.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "use_age_specific": bool(table["use_age_specific"]),
        "crude_annual_rate": _json_safe(table.get("crude_annual_rate")),
        "age_specific": {
            label: _json_safe(value)
            for label, value in table.get("age_specific", {}).items()
        },
        "source": table.get("source"),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)


def validate_mortality_table(
    table: dict[str, Any],
    *,
    strict: bool = False,
) -> list[str]:
    """Check the structure of a mortality table.

    Verification performed:

    1. Round-trips the table through the Pydantic ``MortalityFile``
       schema to catch type/structural issues.
    2. Confirms every Poland bucket label is present.
    3. Confirms each finite rate is in the open interval ``(0, 1)``
       (rates equal to exactly 0 or 1 are biologically impossible for
       whole-bucket human mortality, so they are flagged).
    4. Confirms the crude rate is within a generous Iran-specific
       range (0.2% - 5% per year).

    Parameters
    ----------
    table
        Mortality table dict to check.
    strict
        If ``True``, raise :class:`ValueError` on the first error
        instead of returning a list of warnings. Use ``strict=True``
        in tests and in the published pipeline; leave ``False`` for
        exploratory work.

    Returns
    -------
    list[str]
        A list of human-readable warnings (empty if the table is
        fully valid).
    """
    warnings: list[str] = []

    # 1. Schema round-trip.
    try:
        MortalityFile.model_validate(table)
    except ValidationError as e:
        msg = f"Schema validation failed: {e}"
        if strict:
            raise ValueError(msg) from e
        warnings.append(msg)

    # 2. Bucket coverage.
    expected = {label for label, _ in POLAND_AGE_BUCKETS}
    actual = set(table.get("age_specific", {}))
    missing = expected - actual
    extra = actual - expected
    if missing:
        msg = f"Missing Poland buckets: {sorted(missing)}"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)
    if extra:
        msg = f"Unexpected buckets in age_specific: {sorted(extra)}"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    # 3. Rate range.
    for label, value in table.get("age_specific", {}).items():
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            msg = f"Bucket {label!r} has a non-finite rate: {value!r}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue
        if not (0.0 < float(value) <= 1.0):
            msg = (
                f"Bucket {label!r} rate {value} is outside (0, 1]; "
                f"whole-bucket human mortality should be in (0, 1]."
            )
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    # 4. Crude rate sanity.
    crude = table.get("crude_annual_rate")
    if crude is not None and math.isfinite(float(crude)):
        if not (_MIN_REASONABLE_CRUDE <= float(crude) <= _MAX_REASONABLE_CRUDE):
            msg = (
                f"Crude rate {crude} is outside the expected Iran range "
                f"[{_MIN_REASONABLE_CRUDE}, {_MAX_REASONABLE_CRUDE}]."
            )
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    return warnings


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_to_reference(
    iran_table: dict[str, Any],
    reference_path: str | Path,
) -> pd.DataFrame:
    """Side-by-side comparison of Iran vs a reference mortality file.

    Returns a DataFrame with one row per Poland bucket, columns
    ``iran_per_1000``, ``reference_per_1000``, and the ratio
    ``iran_per_1000 / reference_per_1000``. Buckets missing in
    either side yield ``NaN``.
    """
    ref = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    iran_buckets = iran_table.get("age_specific", {})
    ref_buckets = ref.get("age_specific", {})
    rows: list[dict[str, Any]] = []
    for label, _ in POLAND_AGE_BUCKETS:
        iran_rate = iran_buckets.get(label)
        ref_rate = ref_buckets.get(label)
        rows.append(
            {
                "age_bucket": label,
                "iran_per_1000": (
                    float(iran_rate) * 1000.0
                    if _is_finite_number(iran_rate)
                    else math.nan
                ),
                "reference_per_1000": (
                    float(ref_rate) * 1000.0
                    if _is_finite_number(ref_rate)
                    else math.nan
                ),
                "ratio_iran_over_ref": _safe_ratio(iran_rate, ref_rate),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float:
    """Convert a numeric value to a Python ``float``.

    Handles the mismatch between pandas (which sometimes returns
    Python ``float`` and sometimes ``numpy.float64`` from
    ``.loc[x]`` / ``.iloc[i]``) by trying ``.item()`` first and
    falling back to ``float()``. The ``try`` is defensive against
    types like ``Decimal`` that have ``.item`` but raise.
    """
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    return float(value)


def _json_safe(value: Any) -> float | None:
    """Convert ``NaN``/``inf``/non-numeric to ``None`` (JSON ``null``)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _is_finite_number(value: Any) -> bool:
    """True if ``value`` is a finite int or float (not ``None``/``NaN``)."""
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    """Compute ``numerator / denominator`` if both are finite and the
    denominator is non-zero; otherwise return ``NaN``."""
    if not (_is_finite_number(numerator) and _is_finite_number(denominator)):
        return math.nan
    denom = float(denominator)
    if denom == 0.0:
        return math.nan
    return float(numerator) / denom


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iran-mortality",
        description=(
            "Build an Iran age-specific mortality table matching the "
            "schema of data/mortality.json (Poland). Three modes:\n"
            "  - cohort:  q(x) = 1 - P(x+1, t+1) / P(x, t) from two\n"
            "             Iran_<year>.csv population snapshots.\n"
            "  - wpp:     read m(x, n) directly from a UN Data Portal\n"
            "             abridged-life-table export (recommended).\n"
            "  - merge:   cohort first, then fill null buckets from WPP.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--from-wpp",
        dest="from_wpp",
        action="store_true",
        help="Read rates directly from a UN WPP abridged-life-table export.",
    )
    src.add_argument(
        "--merge-wpp",
        dest="merge_wpp",
        action="store_true",
        help=(
            "Compute the cohort table first, then overwrite any null "
            "buckets with values from the UN WPP export."
        ),
    )
    parser.add_argument(
        "--wpp-csv",
        type=Path,
        default=Path("data/raw/population-un-data-portal-iran.csv"),
        help="Path to the UN Data Portal Iran mortality export.",
    )
    parser.add_argument(
        "--wpp-year",
        type=int,
        default=2024,
        help="Year of the WPP estimate (default: 2024, latest non-projection).",
    )
    parser.add_argument(
        "--wpp-sex",
        choices=("Male", "Female", "Both sexes"),
        default="Male",
        help="Sex to read from the WPP export (default: Male; "
             "hemophilia is X-linked so the modeled cohort is male).",
    )
    parser.add_argument(
        "--year-t",
        type=int,
        default=None,
        help="First cohort year (e.g. 2023). Required for cohort / merge modes.",
    )
    parser.add_argument(
        "--year-t1",
        type=int,
        default=None,
        help="Second cohort year (e.g. 2024). Required for cohort / merge modes.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing Iran_<year>.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mortality_iran.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/mortality.json"),
        help="Optional reference mortality JSON for a side-by-side comparison.",
    )
    parser.add_argument(
        "--sex",
        choices=("total", "male", "female"),
        default="male",
        help="Sex for the cohort method (default: male).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on any validation warning instead of just printing them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code (0 on success)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    table = _run(args)
    write_mortality_json(table, args.output)
    logger.info("Wrote %s", args.output)

    warnings = validate_mortality_table(table, strict=args.strict)
    if warnings:
        logger.warning("Validation warnings:")
        for w in warnings:
            logger.warning("  - %s", w)

    _print_summary(table)
    if args.reference and args.reference.exists():
        _print_comparison(table, args.reference)
    return 0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch to the chosen method and return the table dict."""
    if args.from_wpp or args.merge_wpp:
        if not args.wpp_csv.exists():
            raise FileNotFoundError(
                f"Missing WPP export: {args.wpp_csv}. Re-download from "
                f"the UN Data Portal (Iran, abridged life table, Male, "
                f"{args.wpp_year}, Median, Model-based Estimates) and "
                f"save to that path."
            )
        pop_csv: Path | None = args.data_dir / f"Iran_{args.wpp_year}.csv"
        if pop_csv is not None and not pop_csv.exists():
            logger.warning(
                "Population CSV %s missing; 90+ will use 95-99 fallback",
                pop_csv,
            )
            pop_csv = None
        if args.from_wpp:
            return build_mortality_from_wpp(
                args.wpp_csv,
                year=args.wpp_year,
                sex=args.wpp_sex,
                pop_csv=pop_csv,
            )
        # merge-wpp
        if args.year_t is None or args.year_t1 is None:
            raise SystemExit("--merge-wpp requires --year-t and --year-t1")
        csv_t = args.data_dir / f"Iran_{args.year_t}.csv"
        csv_t1 = args.data_dir / f"Iran_{args.year_t1}.csv"
        cohort = build_mortality_table(csv_t, csv_t1, sex=args.sex)
        merged = merge_cohort_with_wpp(
            cohort,
            args.wpp_csv,
            year=args.wpp_year,
            sex=args.wpp_sex,
            pop_csv=pop_csv,
        )
        filled = merged["source"].get("wpp_filled", [])
        logger.info("Filled %d null buckets from WPP: %s", len(filled), filled)
        return merged

    # Default: cohort method.
    if args.year_t is None or args.year_t1 is None:
        raise SystemExit(
            "Either --from-wpp, --merge-wpp, or both --year-t and "
            "--year-t1 are required."
        )
    csv_t = args.data_dir / f"Iran_{args.year_t}.csv"
    csv_t1 = args.data_dir / f"Iran_{args.year_t1}.csv"
    if not csv_t.exists():
        raise FileNotFoundError(f"Missing {csv_t}")
    if not csv_t1.exists():
        raise FileNotFoundError(f"Missing {csv_t1}")
    table = build_mortality_table(csv_t, csv_t1, sex=args.sex)
    _warn_unreliable_cohort_ages(csv_t, csv_t1, args.sex)
    return table


def _warn_unreliable_cohort_ages(
    csv_t: str | Path, csv_t1: str | Path, sex: Sex
) -> None:
    """Print a warning listing single-year ages where the cohort grew."""
    pop_t = load_iran_population(csv_t)
    pop_t1 = load_iran_population(csv_t1)
    col = {"total": "Total", "male": "M", "female": "F"}[sex]
    s_t = pop_t[col].astype(float)
    s_t1 = pop_t1[col].astype(float)
    common = s_t.index.intersection(s_t1.index)
    s_t, s_t1 = s_t.loc[common], s_t1.loc[common]
    single_year = compute_cohort_mortality(s_t, s_t1)
    unreliable = sorted(
        int(a) for a in single_year.index if not math.isfinite(single_year.loc[a])
    )
    if unreliable:
        logger.warning(
            "Cohort method could not estimate %d single-year ages "
            "(P(x+1, t+1) > P(x, t) - net in-migration or birth pulse). "
            "Their buckets will be null; re-run with --merge-wpp to fill "
            "from UN WPP. Unreliable ages: %s",
            len(unreliable),
            unreliable,
        )


def _print_summary(table: dict[str, Any]) -> None:
    """Print the per-bucket mortality table in per-1000 units."""
    print("\nIran age-specific mortality (per 1000 person-years):")
    for label, _ in POLAND_AGE_BUCKETS:
        rate = table["age_specific"].get(label)
        per_1000 = (
            f"{float(rate) * 1000:.3f}"
            if _is_finite_number(rate)
            else "  n/a"
        )
        print(f"  {label:>7}: {per_1000}")
    crude = table.get("crude_annual_rate")
    if _is_finite_number(crude):
        crude_str = f"{_to_float(crude) * 1000:.3f}"
    else:
        crude_str = "  n/a"
    print(f"  crude   : {crude_str} per 1000 person-years")


def _print_comparison(
    table: dict[str, Any], reference: str | Path
) -> None:
    """Print a side-by-side comparison vs a reference mortality file."""
    print(f"\nComparison vs {reference}:")
    comparison = compare_to_reference(table, reference)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(
            comparison.to_string(
                index=False,
                formatters={
                    "iran_per_1000": _format_optional,
                    "reference_per_1000": "{:.3f}".format,
                    "ratio_iran_over_ref": _format_optional,
                },
                na_rep="  n/a",
            )
        )


def _format_optional(value: float) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "  n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
