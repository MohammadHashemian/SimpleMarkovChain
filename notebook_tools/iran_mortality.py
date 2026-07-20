"""Iran mortality calculator.

Two ways to populate the ``data/mortality.json`` (Poland) schema with
Iran-specific rates:

1. **Cohort survival from population snapshots** (``Iran_<year>.csv``) -
   the original method. Works well for older ages but returns ``null``
   for any age cohort that grew between the two snapshots (Iran's
   young-adult cohorts are still expanding, so the 0-39 buckets come
   out null). See :func:`build_mortality_table`.

2. **UN World Population Prospects (WPP) abridged life table** - reads
   the file exported from the UN Data Portal and emits the same schema
   directly. Recommended for the hemophilia model: hemophilia is
   X-linked recessive and severe disease is essentially male-only, so
   the Male mortality curve is the right input. See
   :func:`build_mortality_from_wpp` and :func:`merge_cohort_with_wpp`.

The two can be combined: use the cohort method where it works (old
ages) and fill the null buckets from WPP (young ages and any other
unreliable age). See :func:`merge_cohort_with_wpp`.

All functions emit rates in the same units as ``mortality.json``:
deaths per person-year (so 0.00369 means 3.69 per 1000 person-years).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

Sex = Literal["total", "male", "female"]


def load_iran_population(path: str | Path) -> pd.DataFrame:
    """Load an ``Iran_<year>.csv`` file and return a frame indexed by age.

    Parameters
    ----------
    path
        Path to a CSV with columns ``Age, M, F``. The ``Age`` column may
        contain the string ``"100+"`` for the open age group, which is
        coerced to the integer ``100``.

    Returns
    -------
    pandas.DataFrame
        Indexed by integer age (0..100). Columns: ``M``, ``F``, ``Total``.
    """
    df = pd.read_csv(path)
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


def compute_cohort_mortality(
    pop_t: pd.Series,
    pop_t1: pd.Series,
) -> pd.Series:
    """Single-year mortality rate via the cohort survival method.

    For each age ``x`` present in ``pop_t``::

        q(x) = 1 - P(x_next, t + 1) / P(x, t)

    where ``x_next = x + 1`` for ``x < 100`` and ``x_next = 100`` for
    ``x = 100`` (the open age group). Negative values from net
    in-migration outpacing deaths are returned as ``NaN`` (and rendered
    as ``null`` in JSON) rather than clipped to zero - silently
    returning 0 would be misleading because we genuinely cannot
    estimate those rates from population growth alone. Values above one
    are clipped to one.
    """
    rates: dict[int, float] = {}
    for x in pop_t.index:
        next_x = x if x == 100 else x + 1
        if next_x not in pop_t1.index:
            rates[int(x)] = np.nan
            continue
        p_t = float(pop_t.loc[x])
        p_t1 = float(pop_t1.loc[next_x])
        if p_t <= 0:
            rates[int(x)] = np.nan
            continue
        q = 1.0 - p_t1 / p_t
        if q < 0.0:
            # Population grew; we cannot disentangle births/in-migration
            # from zero deaths. Mark as unreliable.
            rates[int(x)] = np.nan
            continue
        # Use pure-Python clamp to avoid Pylance complaint about np.clip
        # inferring q as numpy Scalar (which can be complex).
        rates[int(x)] = min(max(q, 0.0), 1.0)
    return pd.Series(rates, name="mortality_rate").sort_index()


def aggregate_to_poland_buckets(
    rates: pd.Series,
    populations: pd.Series,
) -> dict[str, float]:
    """Aggregate single-year ``q(x)`` to the 5-year buckets of ``mortality.json``.

    Each bucket's rate is the population-weighted mean of its constituent
    single-year rates, so the result reflects the rate actually
    experienced by the cohort in the bucket.
    """
    aggregated: dict[str, float] = {}
    for label, age_range in POLAND_AGE_BUCKETS:
        ages = [a for a in age_range if a in rates.index]
        if not ages:
            aggregated[label] = float("nan")
            continue
        bucket_rates = rates.loc[ages]
        bucket_pops = populations.reindex(ages).fillna(0.0)
        valid = ~bucket_rates.isna()
        if not valid.any() or bucket_pops[valid].sum() <= 0:
            aggregated[label] = float("nan")
            continue
        weighted = (bucket_rates[valid] * bucket_pops[valid]).sum()
        denom = bucket_pops[valid].sum()
        aggregated[label] = _to_float(weighted / denom)
    return aggregated


def compute_crude_annual_rate(
    pop_t: pd.Series,
    pop_t1: pd.Series,
) -> float:
    """Crude annual death rate (deaths per person-year).

    Estimated as ``sum_x max(P_t[x] - P_{t+1}[x_next], 0) / sum_x P_t[x]``.
    """
    deaths = 0.0
    for x in pop_t.index:
        next_x = x if x == 100 else x + 1
        if next_x not in pop_t1.index:
            continue
        deaths += max(_to_float(pop_t.loc[x]) - _to_float(pop_t1.loc[next_x]), 0.0)
    total_pop = _to_float(pop_t.sum())
    if total_pop <= 0:
        return float("nan")
    return deaths / total_pop


def build_mortality_table(
    iran_csv_t: str | Path,
    iran_csv_t1: str | Path,
    sex: Sex = "total",
) -> dict[str, Any]:
    """Build an age-specific mortality table from two Iran population snapshots.

    Parameters
    ----------
    iran_csv_t, iran_csv_t1
        Paths to two consecutive ``Iran_<year>.csv`` files. The years
        need not be literally consecutive - any two years will work -
        but closer years give a cleaner single-year cohort estimate.
    sex
        ``"total"`` (default), ``"male"``, or ``"female"``.

    Returns
    -------
    dict
        A dict matching the ``data/mortality.json`` schema with extra
        ``source`` provenance keys.
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
        },
    }


def write_mortality_json(table: dict[str, Any], output: str | Path) -> None:
    """Write the mortality table as JSON to ``output``.

    ``NaN`` values in ``age_specific`` are emitted as JSON ``null`` so
    downstream parsers can detect the unreliable (population grew
    between snapshots) buckets and fill them in from another source.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        "use_age_specific": table["use_age_specific"],
        "crude_annual_rate": _json_safe(table["crude_annual_rate"]),
        "age_specific": {
            label: _json_safe(value) for label, value in table["age_specific"].items()
        },
        "source": table["source"],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)


def _to_float(value: Any) -> float:
    """Convert a numeric value (Python ``float``, ``int``, or a
    numpy/pandas ``Scalar``) to a Python ``float``.

    Pandas sometimes returns Python ``float`` from ``.loc[x]`` /
    ``.iloc[i]`` and sometimes returns a ``numpy.float64``; the
    ``float()`` builtin handles both at runtime, but Pylance's stub
    for ``float.__new__`` rejects ``Scalar`` because the numpy
    ``Scalar`` union includes ``complex``. We unwrap numpy/pandas
    scalars with ``.item()`` first to satisfy the type checker.
    """
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (AttributeError, TypeError):
            pass
    return float(value)


def _json_safe(value: Any) -> float | None:
    """Convert ``NaN``/``inf`` to ``None`` (rendered as JSON ``null``).

    Accepts any numeric type (including numpy/pandas Scalars); the
    ``try`` handles strings and other non-convertible inputs.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def unreliable_single_year_ages(table: dict[str, Any]) -> list[int]:
    """Ages whose single-year rate could not be estimated from the cohort method.

    The single-year rates are not stored in the table; we recompute them
    on the fly to identify which ages need external sources.
    """
    # We can't recover the single-year rates from the bucketed table,
    # so this helper exists for completeness; the CLI computes them
    # directly and passes them in.
    return []


# ---------------------------------------------------------------------------
# UN World Population Prospects (WPP) loader
# ---------------------------------------------------------------------------
#
# The UN Data Portal exports the abridged life table
# (``IndicatorId 79``) with one row per (Sex, AgeStart, Time). The age
# groups match the Poland schema directly up to 85-89; the 90+ bucket
# is split into 90-94, 95-99, and 100+ and is combined here.
#
# Recommended re-download (one time only):
#   Location  : Iran (Islamic Republic of)
#   Indicator : Age specific mortality rate m(x,n) - abridged
#   Sex       : Male (X-linked disease; the modeled cohort is male)
#   Time      : 2024 (latest non-projection; 2025+ are projections)
#   Variant   : Median
#   EstimateType: Model-based Estimates (excludes Projection)

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
    # 90, 95, 100 are combined into the "90+" bucket below.
}

WPP_OPEN_AGE_STARTS: tuple[int, ...] = (90, 95, 100)


def load_wpp_mortality(
    csv_path: str | Path,
    year: int,
    sex: str = "Male",
    indicator_id: int = 79,
) -> pd.DataFrame:
    """Load a UN WPP abridged-life-table export and return one row per
    5-year age group.

    Parameters
    ----------
    csv_path
        Path to the CSV downloaded from the UN Data Portal. Expected
        columns: ``IndicatorId, IndicatorName, ..., Time, Sex,
        AgeStart, AgeEnd, Age, EstimateType, Value``.
    year
        Calendar year to filter on. Projections (EstimateTypeId = 3)
        are excluded so passing a year like 2024 returns observed
        estimates only.
    sex
        ``"Male"``, ``"Female"``, or ``"Both sexes"``.
    indicator_id
        ``79`` for the abridged (5-year) life table, ``80`` for the
        complete (single-year) life table. Abridged is what matches
        the Poland schema.

    Returns
    -------
    pandas.DataFrame
        Indexed by WPP ``AgeStart`` (0, 1, 5, ..., 100). Columns:
        ``age_label`` (the UN's human-readable bucket), ``value`` (the
        mortality rate, deaths per person-year), ``poland_label`` (the
        matching Poland bucket).
    """
    df = pd.read_csv(csv_path)
    if "AgeStart" not in df.columns or "Value" not in df.columns:
        raise ValueError(
            f"{csv_path} doesn't look like a UN Data Portal export "
            "(missing AgeStart / Value columns)."
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
            f"No rows in {csv_path} match "
            f"IndicatorId={indicator_id}, Time={year}, Sex={sex!r}, "
            "Variant=Median, EstimateType=Model-based Estimates. "
            "Did you forget to expand Sex / Age on the UN Data Portal?"
        )
    filtered["AgeStart"] = filtered["AgeStart"].astype(int)
    filtered["Value"] = filtered["Value"].astype(float)
    filtered["poland_label"] = filtered["AgeStart"].map(WPP_AGE_START_TO_POLAND)
    # Rows for 90, 95, 100 -> all collapse to "90+"
    filtered.loc[
        filtered["AgeStart"].isin(WPP_OPEN_AGE_STARTS), "poland_label"
    ] = "90+"
    return (
        filtered.set_index("AgeStart")[["Age", "Value", "poland_label"]]
        .rename(columns={"Age": "age_label", "Value": "value"})
        .sort_index()
    )


def _combine_90_plus(
    wpp: pd.DataFrame,
    pop_csv: str | Path | None,
    sex: str = "Male",
) -> float | None:
    """Combine the 90-94, 95-99 and 100+ WPP buckets into a single
    population-weighted rate for the ``"90+"`` Poland bucket.

    If ``pop_csv`` is provided we use the Iran population snapshot for
    weighting (Male population in each 5-year age range). If not, we
    fall back to the highest-bucket rate (100+) which is a conservative
    (lower-bound) choice for the combined rate.
    """
    open_rows = wpp.loc[wpp.index.isin(WPP_OPEN_AGE_STARTS)]
    if open_rows.empty:
        return None
    if pop_csv is None:
        # Fallback: take the 95-99 rate (mid-range) as a single number.
        if 95 in open_rows.index:
            return _to_float(open_rows.loc[95, "value"])
        return _to_float(open_rows["value"].iloc[0])

    pop = load_iran_population(pop_csv)
    col = {"Male": "M", "Female": "F"}.get(sex, "Total")
    if col not in pop.columns:
        col = "Total"
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


def build_mortality_from_wpp(
    wpp_csv: str | Path,
    year: int,
    sex: str = "Male",
    pop_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build an age-specific mortality table directly from a UN WPP
    export, in the ``data/mortality.json`` schema.

    Parameters
    ----------
    wpp_csv
        Path to the UN Data Portal export.
    year
        Calendar year of the estimate.
    sex
        ``"Male"`` (default - recommended for hemophilia models),
        ``"Female"``, or ``"Both sexes"``.
    pop_csv
        Optional path to an ``Iran_<year>.csv`` file used to
        population-weight the 90-94 / 95-99 / 100+ buckets into a
        single ``"90+"`` rate.
    """
    wpp = load_wpp_mortality(wpp_csv, year=year, sex=sex)
    aggregated: dict[str, float] = {}
    for label, _ in POLAND_AGE_BUCKETS:
        if label == "90+":
            combined = _combine_90_plus(wpp, pop_csv, sex=sex)
            aggregated[label] = float("nan") if combined is None else combined
            continue
        match = wpp[wpp["poland_label"] == label]
        if match.empty:
            aggregated[label] = float("nan")
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
        },
    }


def _crude_annual_rate_from_wpp(
    wpp: pd.DataFrame,
    pop_csv: str | Path | None,
    sex: str,
) -> float:
    """Crude annual death rate from the WPP rates + (optional) population."""
    if pop_csv is None:
        return float("nan")
    pop = load_iran_population(pop_csv)
    col = {"Male": "M", "Female": "F"}.get(sex, "Total")
    if col not in pop.columns:
        col = "Total"
    # Map each WPP bucket to its population
    deaths = 0.0
    total_pop = 0.0
    for age_start in wpp.index:
        # wpp.index is integer because load_wpp_mortality casts AgeStart
        # with .astype(int); iterating it directly avoids the Hashable
        # typing that iterrows() produces.
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
        return float("nan")
    return float(deaths) / float(total_pop)


def merge_cohort_with_wpp(
    cohort_table: dict[str, Any],
    wpp_csv: str | Path,
    year: int,
    sex: str = "Male",
    pop_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Fill any ``None``/``NaN``/missing buckets in ``cohort_table`` with
    rates from the UN WPP export.

    The returned dict is a new table with the same schema; cohort-derived
    values that are present are kept as-is, and only the missing buckets
    are overwritten. The ``source`` block gains a ``wpp_filled`` list of
    bucket labels that were replaced.
    """
    wpp_table = build_mortality_from_wpp(wpp_csv, year=year, sex=sex, pop_csv=pop_csv)
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
            isinstance(current, float) and not np.isfinite(current)
        )
        wpp_value = wpp_table["age_specific"].get(label)
        if is_missing and wpp_value is not None and np.isfinite(wpp_value):
            merged["age_specific"][label] = wpp_value
            filled.append(label)
    merged["source"]["wpp_filled"] = filled
    merged["source"]["wpp_csv"] = str(wpp_csv)
    merged["source"]["wpp_year"] = int(year)
    return merged


def compare_to_reference(
    iran_table: dict[str, Any],
    reference_path: str | Path,
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame (Iran vs reference)."""
    ref = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    iran_buckets = iran_table["age_specific"]
    ref_buckets = ref["age_specific"]
    labels = [label for label, _ in POLAND_AGE_BUCKETS]
    rows: list[dict[str, Any]] = []
    for label in labels:
        iran_rate = iran_buckets.get(label)
        ref_rate = ref_buckets.get(label)
        ratio = (
            float(iran_rate) / float(ref_rate)
            if iran_rate is not None
            and ref_rate not in (None, 0)
            and not np.isnan(float(iran_rate or "nan"))
            and not np.isnan(float(ref_rate or "nan"))
            else float("nan")
        )
        rows.append(
            {
                "age_bucket": label,
                "iran_per_1000": float(iran_rate) * 1000.0
                if iran_rate is not None
                else float("nan"),
                "reference_per_1000": float(ref_rate) * 1000.0
                if ref_rate is not None
                else float("nan"),
                "ratio_iran_over_ref": ratio,
            }
        )
    return pd.DataFrame(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an Iran age-specific mortality table, matching the "
            "schema of data/mortality.json (Poland). Either compute it "
            "from two consecutive population snapshots (cohort method), "
            "or read it directly from a UN WPP abridged-life-table export."
        )
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--from-wpp",
        dest="from_wpp",
        action="store_true",
        help=(
            "Read rates directly from a UN World Population Prospects "
            "abridged-life-table export (--wpp-csv)."
        ),
    )
    src.add_argument(
        "--merge-wpp",
        dest="merge_wpp",
        action="store_true",
        help=(
            "Compute the cohort table first, then overwrite any null "
            "buckets with values from the UN WPP export (--wpp-csv)."
        ),
    )
    parser.add_argument(
        "--wpp-csv",
        type=Path,
        default=Path("data/raw/population-un-data-portal-iran.csv"),
        help="Path to the UN Data Portal Iran mortality export",
    )
    parser.add_argument(
        "--wpp-year",
        type=int,
        default=2024,
        help="Year of the WPP estimate to use (default: 2024, latest non-projection)",
    )
    parser.add_argument(
        "--wpp-sex",
        choices=("Male", "Female", "Both sexes"),
        default="Male",
        help="Sex to read from the WPP export (default: Male - hemophilia is male-only)",
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
        help="Directory containing Iran_<year>.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mortality_iran.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/mortality.json"),
        help="Optional reference mortality JSON for a side-by-side comparison",
    )
    parser.add_argument(
        "--sex",
        choices=("total", "male", "female"),
        default="male",
        help="Sex for the cohort method (default: male)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    if args.from_wpp or args.merge_wpp:
        if not args.wpp_csv.exists():
            raise FileNotFoundError(
                f"Missing WPP export: {args.wpp_csv}. Re-download from the "
                "UN Data Portal (Iran, abridged life table, Male, 2024, Median, "
                "Model-based Estimates) and save to that path."
            )
        pop_csv: Path | None = args.data_dir / f"Iran_{args.wpp_year}.csv"
        if pop_csv is not None and not pop_csv.exists():
            logger.warning("Population CSV %s missing; 90+ will use 95-99 fallback", pop_csv)
            pop_csv = None
        if args.from_wpp:
            table = build_mortality_from_wpp(
                args.wpp_csv, year=args.wpp_year, sex=args.wpp_sex, pop_csv=pop_csv
            )
        else:
            if args.year_t is None or args.year_t1 is None:
                raise SystemExit("--merge-wpp requires --year-t and --year-t1")
            csv_t = args.data_dir / f"Iran_{args.year_t}.csv"
            csv_t1 = args.data_dir / f"Iran_{args.year_t1}.csv"
            cohort = build_mortality_table(csv_t, csv_t1, sex=args.sex)
            table = merge_cohort_with_wpp(
                cohort,
                args.wpp_csv,
                year=args.wpp_year,
                sex=args.wpp_sex,
                pop_csv=pop_csv,
            )
            filled = table["source"].get("wpp_filled", [])
            print(
                f"\nFilled {len(filled)} null buckets from WPP: {filled}"
            )
    else:
        if args.year_t is None or args.year_t1 is None:
            raise SystemExit("Either --from-wpp, --merge-wpp, or both --year-t and --year-t1 are required")
        csv_t = args.data_dir / f"Iran_{args.year_t}.csv"
        csv_t1 = args.data_dir / f"Iran_{args.year_t1}.csv"
        if not csv_t.exists():
            raise FileNotFoundError(f"Missing {csv_t}")
        if not csv_t1.exists():
            raise FileNotFoundError(f"Missing {csv_t1}")

        pop_t_df = load_iran_population(csv_t)
        pop_t1_df = load_iran_population(csv_t1)
        if args.sex == "total":
            s_t, s_t1 = pop_t_df["Total"], pop_t1_df["Total"]
        elif args.sex == "male":
            s_t, s_t1 = pop_t_df["M"], pop_t1_df["M"]
        else:
            s_t, s_t1 = pop_t_df["F"], pop_t1_df["F"]
        common = s_t.index.intersection(s_t1.index)
        s_t, s_t1 = s_t.loc[common].astype(float), s_t1.loc[common].astype(float)

        single_year_rates = compute_cohort_mortality(s_t, s_t1)
        unreliable = sorted(
            int(a) for a in single_year_rates.index if np.isnan(single_year_rates.loc[a])
        )

        table = build_mortality_table(csv_t, csv_t1, sex=args.sex)

        if unreliable:
            print(
                "\nWARNING: the following single-year ages had a growing cohort"
                "\n         (P(x+1, t+1) > P(x, t)) so the cohort survival method"
                "\n         cannot estimate mortality. Their buckets will be null"
                "\n         in the JSON; re-run with --merge-wpp to fill from UN WPP:"
                f"\n         {unreliable}"
            )

    write_mortality_json(table, args.output)
    logger.info("Wrote %s", args.output)

    print("\nIran age-specific mortality (per 1000 person-years):")
    for label, _ in POLAND_AGE_BUCKETS:
        rate = table["age_specific"].get(label)
        per_1000 = (
            f"{rate * 1000:.3f}" if rate is not None and not np.isnan(rate) else "  n/a"
        )
        print(f"  {label:>7}: {per_1000}")
    crude = table["crude_annual_rate"]
    crude_str = f"{crude * 1000:.3f}" if crude is not None and not np.isnan(crude) else "  n/a"
    print(f"  crude   : {crude_str} per 1000 person-years")

    if args.reference and args.reference.exists():
        print(f"\nComparison vs {args.reference}:")
        comparison = compare_to_reference(table, args.reference)
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

    return 0


def _format_optional(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "  n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
