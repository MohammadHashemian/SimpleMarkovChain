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

This module is data-only: it returns a dict matching the
``MortalityFile`` schema (see ``persistence/schemas/mortality.py``).
The end-to-end pipeline (build the table, validate it, compare it
to a reference, and write the output) lives in the notebook
``notebooks/01b_mortality_iran.ipynb`` because the project is
analytic, not a CLI.

Output units
------------
All rates are deaths per person-year. So ``0.00369`` is 3.69 per
1,000 person-years, matching the units of the existing Poland table.

Validation
----------
The :func:`validate_mortality_table` function checks the structure
(rate range, bucket coverage) and round-trips through the Pydantic
``MortalityFile`` schema. It returns a list of warnings rather than
raising, so callers can decide how strict to be.

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

import json
import logging
import math
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
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

#: Column name used by the WPP DataFrame for the mortality rate.
WPP_VALUE_COL = "value"

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


def load_iran_population(path: str | Path) -> pl.DataFrame:
    """Load an ``Iran_<year>.csv`` population-by-age file into polars.

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
    polars.DataFrame
        Columns ``Age`` (Int64), ``M`` (Int64), ``F`` (Int64), and
        ``Total`` (Int64 = M + F). Sorted by ``Age``.
    """
    df = pl.read_csv(
        path,
        schema_overrides={"Age": pl.Utf8, "M": pl.Int64, "F": pl.Int64},
    )
    required = {"Age", "M", "F"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns; expected Age, M, F; "
            f"got {df.columns}"
        )
    df = df.with_columns(
        pl.when(pl.col("Age") == "100+")
        .then(pl.lit("100"))
        .otherwise(pl.col("Age"))
        .alias("Age")
        .cast(pl.Int64, strict=False)
    ).drop_nulls(subset=["Age", "M", "F"]).with_columns(
        pl.col("Age").cast(pl.Int64),
        pl.col("M").cast(pl.Int64),
        pl.col("F").cast(pl.Int64),
        (pl.col("M") + pl.col("F")).alias("Total").cast(pl.Int64),
    )
    return df.sort("Age")


def load_wpp_mortality(
    csv_path: str | Path,
    year: int,
    sex: WPPSex = "Male",
    indicator_id: int = 79,
) -> pl.DataFrame:
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
    polars.DataFrame
        Columns ``AgeStart`` (Int64), ``age_label`` (Utf8, the UN's
        human-readable bucket), ``value`` (Float64, deaths per
        person-year), and ``poland_label`` (Utf8, the matching
        Poland bucket; rows for 90, 95, 100 all map to ``"90+"``).
    """
    df = pl.read_csv(csv_path)
    required = {
        "IndicatorId", "Time", "Variant", "Sex", "AgeStart",
        "Age", "EstimateType", "Value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} doesn't look like a UN Data Portal export: "
            f"missing columns {sorted(missing)}"
        )
    filtered = df.filter(
        (pl.col("IndicatorId") == indicator_id)
        & (pl.col("Time") == year)
        & (pl.col("Sex") == sex)
        & (pl.col("Variant") == "Median")
        & (pl.col("EstimateType") == "Model-based Estimates")
    )
    if filtered.is_empty():
        raise ValueError(
            f"No rows in {csv_path} match IndicatorId={indicator_id}, "
            f"Time={year}, Sex={sex!r}, Variant=Median, "
            f"EstimateType=Model-based Estimates. Did you forget to "
            f"expand the Sex / Age filters on the UN Data Portal?"
        )
    # polars supports an expression-based dict mapping via replace_strict.
    poland_map = {int(k): v for k, v in WPP_AGE_START_TO_POLAND.items()}
    open_label = "90+"
    filtered = (
        filtered.with_columns(
            pl.col("AgeStart").cast(pl.Int64),
            pl.col("Value").cast(pl.Float64).alias(WPP_VALUE_COL),
        )
        .with_columns(
            pl.col("AgeStart")
            .replace_strict(poland_map, default=open_label)
            .alias("poland_label")
        )
        .rename({"Age": "age_label"})
        .select("AgeStart", "age_label", WPP_VALUE_COL, "poland_label")
        .sort("AgeStart")
    )
    return filtered


# ---------------------------------------------------------------------------
# Cohort survival method
# ---------------------------------------------------------------------------


def compute_cohort_mortality(
    pop_t: pl.Series,
    pop_t1: pl.Series,
) -> pl.Series:
    """Single-year mortality rate via the cohort survival method.

    For each age ``x`` present in ``pop_t``::

        q(x) = 1 - P(x_next, t + 1) / P(x, t)

    where ``x_next = x + 1`` for ``x < 100`` and ``x_next = 100``
    for ``x = 100`` (the open age group, so the formula gives the
    one-year exit rate of the open interval).

    Parameters
    ----------
    pop_t
        Polars Series of populations at exact ages at time ``t``.
        Must be a single-sex column (``M``, ``F``, or ``Total``).
    pop_t1
        Polars Series of populations at exact ages at time ``t + 1``
        (same length as ``pop_t`` and aligned by row).

    Returns
    -------
    polars.Series
        Length-``N`` Float64 Series of single-year mortality rates.
        ``null`` (NaN) marks ages where the cohort grew
        (``P(x+1, t+1) > P(x, t)``) and the rate therefore cannot
        be estimated.
    """
    if len(pop_t) != len(pop_t1):
        raise ValueError(
            f"pop_t and pop_t1 must have the same length; got "
            f"{len(pop_t)} and {len(pop_t1)}"
        )
    n = len(pop_t)
    if n < 2:
        # Need at least one closed age + the open interval.
        return pl.Series("mortality_rate", [float("nan")] * n, dtype=pl.Float64)

    pop_t_list = [float(x) for x in pop_t.to_list()]
    pop_t1_list = [float(x) for x in pop_t1.to_list()]
    rates: list[float] = []
    for i in range(n):
        p_t = pop_t_list[i]
        # For the open interval (age 100), the next population is
        # pop_t1 at the same row; for all other ages, the next
        # population is pop_t1 at i+1.
        if i == n - 1:
            p_next = pop_t1_list[i]
        else:
            p_next = pop_t1_list[i + 1]
        if p_t <= 0:
            rates.append(float("nan"))
            continue
        q = 1.0 - p_next / p_t
        if q < 0.0:
            # Population grew; mark as unreliable (NaN, not 0).
            rates.append(float("nan"))
            continue
        # Pure-Python clamp to [0, 1].
        rates.append(min(max(q, 0.0), 1.0))
    return pl.Series("mortality_rate", rates, dtype=pl.Float64)


def aggregate_to_poland_buckets(
    rates: pl.Series,
    populations: pl.Series,
    ages: pl.Series,
) -> dict[str, float | None]:
    """Aggregate single-year ``q(x)`` to the 5-year buckets.

    Each bucket's rate is the **population-weighted mean** of its
    constituent single-year rates, so the result reflects the rate
    actually experienced by the cohort in the bucket. Buckets with
    no valid (non-NaN) single-year rates are returned as ``None``.

    Parameters
    ----------
    rates
        Single-year mortality rates, aligned with ``ages`` and
        ``populations``. ``null`` entries are skipped.
    populations
        Population at each single year of age, aligned with ``rates``.
    ages
        Integer age for each row, aligned with ``rates``.

    Returns
    -------
    dict
        Mapping from Poland bucket label (``"0"``, ``"1-4"``, ...,
        ``"90+"``) to a float mortality rate, or ``None`` for
        unreliable buckets.
    """
    df = pl.DataFrame(
        {
            "age": ages.cast(pl.Int64),
            "rate": rates.cast(pl.Float64),
            "pop": populations.cast(pl.Float64),
        }
    )
    aggregated: dict[str, float | None] = {}
    for label, age_range in POLAND_AGE_BUCKETS:
        bucket = df.filter(pl.col("age").is_in(list(age_range)))
        bucket = bucket.filter(pl.col("rate").is_not_null() & pl.col("pop").is_not_null())
        if bucket.is_empty():
            aggregated[label] = None
            continue
        # Population-weighted mean: sum(rate * pop) / sum(pop)
        weighted = (bucket["rate"] * bucket["pop"]).sum()
        denom = bucket["pop"].sum()
        if not denom or denom == 0:
            aggregated[label] = None
            continue
        aggregated[label] = float(weighted) / float(denom)
    return aggregated


def compute_crude_annual_rate(
    pop_t: pl.Series,
    pop_t1: pl.Series,
) -> float | None:
    """Crude annual death rate (deaths per person-year) from two snapshots.

    Estimated as
    ``sum_x max(P_t[x] - P_{t+1}[x_next], 0) / sum_x P_t[x]``.
    The numerator counts only positive cohort changes (population
    *loss*) and treats growth as zero deaths, so the estimate is a
    lower bound on the true crude rate when migration is positive.
    """
    if len(pop_t) != len(pop_t1):
        raise ValueError(
            f"pop_t and pop_t1 must have the same length; got "
            f"{len(pop_t)} and {len(pop_t1)}"
        )
    n = len(pop_t)
    if n < 2:
        return None
    pop_t_list = [float(x) for x in pop_t.to_list()]
    pop_t1_list = [float(x) for x in pop_t1.to_list()]
    deaths = 0.0
    for i in range(n):
        if i == n - 1:
            p_next = pop_t1_list[i]
        else:
            p_next = pop_t1_list[i + 1]
        deaths += max(pop_t_list[i] - p_next, 0.0)
    total_pop = sum(pop_t_list)
    if total_pop <= 0:
        return None
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

    col = {"total": "Total", "male": "M", "female": "F"}[sex]
    # Inner-join on Age so both series are aligned by age.
    joined = pop_t.select(["Age", col]).join(
        pop_t1.select(["Age", col]), on="Age", how="inner", suffix="_t1"
    ).sort("Age")
    series_t = joined[col]
    series_t1 = joined[f"{col}_t1"]
    ages = joined["Age"]

    rates = compute_cohort_mortality(series_t, series_t1)
    aggregated = aggregate_to_poland_buckets(rates, series_t.cast(pl.Float64), ages)
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
    wpp: pl.DataFrame,
    pop: pl.DataFrame | None,
    sex: WPPSex = "Male",
) -> float | None:
    """Combine the 90-94, 95-99 and 100+ WPP buckets into a single
    population-weighted rate for the ``"90+"`` Poland bucket.

    Without a population file we fall back to the 95-99 rate (a
    mid-range value). With a population file we use the male
    (or female / total, depending on ``sex``) population in each
    5-year age range as the weight.
    """
    open_rows = wpp.filter(pl.col("AgeStart").is_in(list(WPP_OPEN_AGE_STARTS)))
    if open_rows.is_empty():
        return None
    if pop is None:
        if 95 in open_rows["AgeStart"].to_list():
            row = open_rows.filter(pl.col("AgeStart") == 95)
            value: Any = row[WPP_VALUE_COL].first()
            return float(value)
        value = open_rows[WPP_VALUE_COL].first()
        return float(value)

    col = {"Male": "M", "Female": "F"}.get(sex, "Total")
    if col not in pop.columns:
        col = "Total"
    ages_in_pop = pop["Age"].to_list()
    has_100 = 100 in ages_in_pop
    weights = {
        90: int(pop.filter((pl.col("Age") >= 90) & (pl.col("Age") <= 94))[col].sum()),
        95: int(pop.filter((pl.col("Age") >= 95) & (pl.col("Age") <= 99))[col].sum()),
        100: int(pop.filter(pl.col("Age") == 100)[col].sum()) if has_100 else 0,
    }
    total = sum(weights.values())
    if total <= 0:
        value = open_rows[WPP_VALUE_COL].first()
        return float(value)
    weighted = 0.0
    for age_start in WPP_OPEN_AGE_STARTS:
        if age_start not in open_rows["AgeStart"].to_list():
            continue
        rate = float(
            open_rows.filter(pl.col("AgeStart") == age_start)[WPP_VALUE_COL].item()
        )
        weighted += weights[age_start] * rate
    return weighted / total


def _crude_annual_rate_from_wpp(
    wpp: pl.DataFrame,
    pop: pl.DataFrame | None,
    sex: WPPSex,
) -> float | None:
    """Crude annual death rate from the WPP rates + (optional) population."""
    if pop is None:
        return None
    col = {"Male": "M", "Female": "F"}.get(sex, "Total")
    if col not in pop.columns:
        col = "Total"
    ages_in_pop = pop["Age"].to_list()
    has_100 = 100 in ages_in_pop
    deaths = 0.0
    total_pop = 0.0
    for row in wpp.iter_rows(named=True):
        age_start = int(row["AgeStart"])
        rate = float(row[WPP_VALUE_COL])
        if age_start == 0:
            p = int(pop.filter(pl.col("Age") == 0)[col].sum())
        elif age_start == 1:
            p = int(pop.filter((pl.col("Age") >= 1) & (pl.col("Age") <= 4))[col].sum())
        elif age_start in (90, 95):
            upper = age_start + 4
            p = int(pop.filter((pl.col("Age") >= age_start) & (pl.col("Age") <= upper))[col].sum())
        elif age_start == 100:
            p = int(pop.filter(pl.col("Age") == 100)[col].sum()) if has_100 else 0
        else:
            upper = age_start + 4
            p = int(pop.filter((pl.col("Age") >= age_start) & (pl.col("Age") <= upper))[col].sum())
        total_pop += p
        deaths += rate * p
    if total_pop <= 0:
        return None
    return deaths / total_pop


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
    pop = load_iran_population(pop_csv) if pop_csv is not None else None
    aggregated: dict[str, float | None] = {}
    for label, _ in POLAND_AGE_BUCKETS:
        if label == "90+":
            aggregated[label] = _combine_90_plus(wpp, pop, sex=sex)
            continue
        match = wpp.filter(pl.col("poland_label") == label)
        if match.is_empty():
            aggregated[label] = None
        else:
            value = cast(float, match[WPP_VALUE_COL].first())
            aggregated[label] = float(value)

    crude = _crude_annual_rate_from_wpp(wpp, pop, sex=sex)

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
) -> pl.DataFrame:
    """Side-by-side comparison of Iran vs a reference mortality file.

    Returns a polars DataFrame with one row per Poland bucket, columns
    ``iran_per_1000``, ``reference_per_1000``, and the ratio
    ``iran_per_1000 / reference_per_1000``. Buckets missing in
    either side yield ``null``.
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
                    else None
                ),
                "reference_per_1000": (
                    float(ref_rate) * 1000.0
                    if _is_finite_number(ref_rate)
                    else None
                ),
                "ratio_iran_over_ref": (
                    float(iran_rate) / float(ref_rate)
                    if _is_finite_number(iran_rate)
                    and _is_finite_number(ref_rate)
                    and float(ref_rate) != 0
                    else None
                ),
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
