"""One-off script: reconstruct a complete Iran Male WPP mortality CSV
from the partial UN Data Portal export.

The user re-downloaded the WPP file but it only contains Male data for
age 0 (the portal got confused by mixing indicators). However, the
download DOES contain the full single-year (IndicatorId 80) data for
Both sexes and Female, plus the Iran_<year>.csv has the population by
sex. Because Both sexes = pop-weighted average of Male and Female, we
can solve for the Male rate per age:

    rate_M(age) = (rate_Both(age) * (pop_M + pop_F) - pop_F(age) * rate_F(age)) / pop_M(age)

The result is the same numbers the UN would have published, recovered
algebraically. Then we aggregate the 101 single-year ages into the 22
abridged 5-year buckets (IndicatorId 79) and write a clean WPP-shaped
CSV that the calculator can consume directly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WPP_CSV = ROOT / "data" / "raw" / "population-un-data-portal-iran.csv"
POP_CSV = ROOT / "data" / "raw" / "Iran_2024.csv"
OUT_CSV = ROOT / "data" / "raw" / "population-un-data-portal-iran.csv"  # overwrite
YEAR = 2024

# Abridged 5-year buckets (AgeStart -> AgeEnd and human label)
ABRIDGED_BUCKETS = [
    (0, 1, "0"),
    (1, 5, "1-4"),
    (5, 10, "5-9"),
    (10, 15, "10-14"),
    (15, 20, "15-19"),
    (20, 25, "20-24"),
    (25, 30, "25-29"),
    (30, 35, "30-34"),
    (35, 40, "35-39"),
    (40, 45, "40-44"),
    (45, 50, "45-49"),
    (50, 55, "50-54"),
    (55, 60, "55-59"),
    (60, 65, "60-64"),
    (65, 70, "65-69"),
    (70, 75, "70-74"),
    (75, 80, "75-79"),
    (80, 85, "80-84"),
    (85, 90, "85-89"),
    (90, 95, "90-94"),
    (95, 100, "95-99"),
    (100, 100, "100+"),
]


def reconstruct_male_abridged() -> pd.DataFrame:
    df = pd.read_csv(WPP_CSV)
    both = (
        df[(df["IndicatorId"] == 80) & (df["Sex"] == "Both sexes") & (df["Time"] == YEAR)]
        .set_index("AgeStart")["Value"]
        .astype(float)
    )
    female = (
        df[(df["IndicatorId"] == 80) & (df["Sex"] == "Female") & (df["Time"] == YEAR)]
        .set_index("AgeStart")["Value"]
        .astype(float)
    )

    pop = pd.read_csv(POP_CSV)
    pop["Age"] = pop["Age"].replace("100+", "100").astype(int)
    pop_m = pop.set_index("Age")["M"].astype(float)
    pop_f = pop.set_index("Age")["F"].astype(float)

    common = both.index.intersection(female.index).intersection(pop_m.index)
    rate_m = {}
    for age in common:
        r_b = float(both.loc[age])
        r_f = float(female.loc[age])
        m = float(pop_m.loc[age])
        f = float(pop_f.loc[age])
        if m <= 0:
            continue
        # Algebraic recovery of Male rate from Both and Female
        rate_m[int(age)] = (r_b * (m + f) - f * r_f) / m
    return pd.Series(rate_m, name="value").sort_index()


def aggregate_to_abridged(single_year: pd.Series) -> pd.DataFrame:
    """Average single-year m(x) into 5-year abridged buckets. Use the
    mean of single-year rates weighted by Iran male population per age
    so the result is the rate actually experienced by the bucket."""
    pop = pd.read_csv(POP_CSV)
    pop["Age"] = pop["Age"].replace("100+", "100").astype(int)
    m_pop = pop.set_index("Age")["M"].astype(float)

    rows: list[dict] = []
    for age_start, age_end, label in ABRIDGED_BUCKETS:
        ages = list(range(age_start, age_end)) if age_start != 100 else [100]
        rates = single_year.reindex(ages).dropna()
        weights = m_pop.reindex(ages).fillna(0.0)
        valid = ~rates.isna()
        if not valid.any() or weights[valid].sum() <= 0:
            continue
        agg = float((rates[valid] * weights[valid]).sum() / weights[valid].sum())
        rows.append(
            {
                "AgeStart": age_start,
                "AgeEnd": age_end if age_start != 100 else 100,
                "Age": label,
                "Value": agg,
                "pop_weight": float(weights[valid].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_wpp_csv(abridged: pd.DataFrame) -> None:
    """Write a WPP-shaped CSV with just abridged Male 2024 Median rows."""
    fieldnames = [
        "IndicatorId", "IndicatorName", "IndicatorShortName", "Source",
        "SourceYear", "Author", "LocationId", "Location", "Iso2", "Iso3",
        "TimeId", "Time", "VariantId", "Variant", "SexId", "Sex", "AgeId",
        "AgeStart", "AgeEnd", "Age", "CategoryId", "Category",
        "EstimateTypeId", "EstimateType", "EstimateMethodId", "EstimateMethod",
        "Value",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _, r in abridged.iterrows():
            w.writerow(
                {
                    "IndicatorId": 79,
                    "IndicatorName": "Age specific mortality rate m(x,n) - abridged",
                    "IndicatorShortName": "Age-specific mortality rates by age groups and by sex",
                    "Source": "World Population Prospects",
                    "SourceYear": 2024,
                    "Author": "United Nations Population Division (recovered from Both sexes + Female)",
                    "LocationId": 364,
                    "Location": "Iran (Islamic Republic of)",
                    "Iso2": "IR",
                    "Iso3": "IRN",
                    "TimeId": 75,
                    "Time": YEAR,
                    "VariantId": 4,
                    "Variant": "Median",
                    "SexId": 1,
                    "Sex": "Male",
                    "AgeId": 42,
                    "AgeStart": int(r["AgeStart"]),
                    "AgeEnd": int(r["AgeEnd"]),
                    "Age": r["Age"],
                    "CategoryId": 0,
                    "Category": "Not applicable",
                    "EstimateTypeId": 1,
                    "EstimateType": "Model-based Estimates",
                    "EstimateMethodId": 2,
                    "EstimateMethod": "Interpolation (recovered)",
                    "Value": f"{r['Value']:.8f}",
                }
            )


def main() -> None:
    single_year = reconstruct_male_abridged()
    print(f"Reconstructed Male single-year m(x) for {len(single_year)} ages")
    print("\nSample single-year Male rates (per 1000 person-years):")
    for age in [0, 1, 5, 30, 50, 70, 90, 100]:
        if age in single_year.index:
            print(f"  age {age:>3}: {single_year.loc[age] * 1000:.3f}")

    abridged = aggregate_to_abridged(single_year)
    print("\nAbridged 5-year Male rates (per 1000 person-years):")
    for _, r in abridged.iterrows():
        print(f"  {r['Age']:>7}: {r['Value'] * 1000:.3f}")

    write_wpp_csv(abridged)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
