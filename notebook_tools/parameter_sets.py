import pickle
from pathlib import Path

import numpy as np

from analysis.distributions import (
    BetaFromMeanSD,
    Constant,
    GammaFromMeanCV,
    TriangularDist,
)
from analysis.psa.models import ParameterSet
from analysis.psa.parameters import Parameter


class HemophiliaParamRepo:
    def __init__(self, root: Path, cache_path: Path):
        self.root = root
        self.cache_path = cache_path
        self.ows_params_keys = [
            "joint_bleeding_fraction",
            "life_threatening_bleeding_fraction",
            "healthy_utility",
            "mild_arthropathy_utility",
            "moderate_arthropathy_utility",
            "severe_arthropathy_utility",
            "spontaneous_bleeding_utility",
            "joint_bleeding_utility",
            "life_threatening_bleeding_utility",
            "prophylaxis_background_factor_consumption_per_kg",
            "factor_consumption_per_spontaneous_bleeding_per_kg",
            "factor_consumption_per_joint_bleeding_per_kg",
            "factor_consumption_per_life_threatening_bleeding_per_kg",
        ]

    def load_psa_parameters(self) -> ParameterSet:
        with open(self.root / self.cache_path, "rb") as f:
            samples = pickle.load(f)
            params = ParameterSet(
                cycles=Parameter(
                    distribution=Constant(value=10 * 52)
                ),  # EARLY SCENARIO
                baseline_age=Parameter(distribution=Constant(value=2)),
                weight_factor=Parameter(distribution=Constant(value=1.0)),
                benefits_discount_rate=Parameter(distribution=Constant(value=0)),
                costs_discount_rate=Parameter(distribution=Constant(value=0)),
                # Clinical
                bleeding_rate=Parameter(
                    distribution=Constant(value=0),
                    cache=samples["on_demand"]["bayesian"],
                ),  # Cache Data from meta_analysis
                joint_bleeding_fraction=Parameter(
                    distribution=BetaFromMeanSD(mean=0.75, sd=0.0255)
                ),
                life_threatening_bleeding_fraction=Parameter(
                    distribution=TriangularDist(left=0.01, mode=0.025, right=0.05)
                ),
                # Benefits
                healthy_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=0.915, cv=0.05)
                ),
                mild_arthropathy_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=0.85, cv=0.05)
                ),
                moderate_arthropathy_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=0.78, cv=0.05)
                ),
                severe_arthropathy_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=0.68, cv=0.05)
                ),
                spontaneous_bleeding_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=0.60, cv=0.05)
                ),
                joint_bleeding_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=0.50, cv=0.05)
                ),
                life_threatening_bleeding_utility=Parameter(
                    distribution=BetaFromMeanSD(mean=0.25, cv=0.05)
                ),
                death_utility=Parameter(distribution=Constant(value=0)),
                # Costs
                per_unit_price=Parameter(distribution=Constant(value=58_000)),
                prophylaxis_background_factor_consumption_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=75, cv=0.5)
                ),
                factor_consumption_per_spontaneous_bleeding_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=120, cv=0.20)
                ),
                factor_consumption_per_joint_bleeding_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=60, cv=0.15)
                ),
                factor_consumption_per_life_threatening_bleeding_per_kg=Parameter(
                    distribution=GammaFromMeanCV(mean=550, cv=0.25)
                ),
            )
        return params

    def load_owsa_parameters(self) -> ParameterSet:

        with open(self.root / self.cache_path, "rb") as f:
            samples = pickle.load(f)
            # on_demand base scenario
            params = ParameterSet(
                cycles=Parameter(Constant(value=10 * 52)),  # EARLY SCENARIO
                baseline_age=Parameter(Constant(value=2)),
                weight_factor=Parameter(Constant(value=1.0)),
                benefits_discount_rate=Parameter(Constant(value=0)),
                costs_discount_rate=Parameter(Constant(value=0)),
                # Clinical
                bleeding_rate=Parameter(
                    Constant(value=np.mean(samples["on_demand"]["bayesian"]))
                ),
                joint_bleeding_fraction=Parameter(Constant(value=0.75)),  # MEAN
                life_threatening_bleeding_fraction=Parameter(
                    Constant(value=0.025)
                ),  # MODE
                # Benefits
                healthy_utility=Parameter(Constant(value=0.915)),  # MEAN
                mild_arthropathy_utility=Parameter(Constant(value=0.85)),  # MEAN
                moderate_arthropathy_utility=Parameter(Constant(value=0.78)),  # MEAN
                severe_arthropathy_utility=Parameter(Constant(value=0.68)),  # MEAN
                spontaneous_bleeding_utility=Parameter(Constant(value=0.60)),  # MEAN
                joint_bleeding_utility=Parameter(Constant(value=0.50)),  # MEAN
                life_threatening_bleeding_utility=Parameter(
                    Constant(value=0.25)
                ),  # MEAN
                death_utility=Parameter(Constant(value=0)),
                # Costs
                per_unit_price=Parameter(distribution=Constant(value=58_000)),
                prophylaxis_background_factor_consumption_per_kg=Parameter(
                    Constant(value=75)
                ),
                factor_consumption_per_spontaneous_bleeding_per_kg=Parameter(
                    Constant(120)
                ),
                factor_consumption_per_joint_bleeding_per_kg=Parameter(
                    Constant(value=60)
                ),
                factor_consumption_per_life_threatening_bleeding_per_kg=Parameter(
                    Constant(value=550)
                ),
            )
        return params
