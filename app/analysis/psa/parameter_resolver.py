
import numpy as np

from app.domain.inputs import ModelInput


class ParameterResolver:
    @staticmethod
    def resolve_samples(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Vectorized deterministic transformation layer
        Handles all dependencies safely.
        """

        bleeding = samples["bleeding_rate"]

        joint_frac = samples["joint_bleeding_fraction"]
        lt_frac = samples["life_threatening_bleeding_fraction"]

        # ---- Safety: enforce valid simplex ----
        # Prevent pathological PSA draws (rare but critical)
        total_frac = joint_frac + lt_frac
        overflow_mask = total_frac > 1.0

        if np.any(overflow_mask):
            # Normalize proportions if they exceed 1
            joint_frac = np.where(overflow_mask, joint_frac / total_frac, joint_frac)
            lt_frac = np.where(overflow_mask, lt_frac / total_frac, lt_frac)

        spontaneous_frac = 1.0 - joint_frac - lt_frac

        # ---- Derived rates ----
        spontaneous_rate = bleeding * spontaneous_frac
        joint_rate = bleeding * joint_frac
        lt_rate = bleeding * lt_frac

        return {
            **samples,
            "spontaneous_bleeding_rate": spontaneous_rate,
            "joint_bleeding_rate": joint_rate,
            "life_threatening_bleeding_rate": lt_rate,
        }

    @staticmethod
    def build_single(res: dict[str, np.ndarray], i: int) -> ModelInput:
        """
        Convert one PSA draw into model-ready structure
        """

        return ModelInput(
            # Time
            cycle=res["cycles"][i],
            # Clinical
            bleeding_rate=res["bleeding_rate"][i],
            spontaneous_bleeding_rate=res["spontaneous_bleeding_rate"][i],
            joint_bleeding_rate=res["joint_bleeding_rate"][i],
            life_threatening_bleeding_rate=res["life_threatening_bleeding_rate"][i],
            # Demographics
            baseline_age=res["baseline_age"][i],
            weight_factor=res["weight_factor"][i],
            # Utilities
            benefits_discount_rate=res["benefits_discount_rate"][i],
            healthy_utility=res["healthy_utility"][i],
            mild_arthropathy_utility=res["mild_arthropathy_utility"][i],
            moderate_arthropathy_utility=res["moderate_arthropathy_utility"][i],
            severe_arthropathy_utility=res["severe_arthropathy_utility"][i],
            spontaneous_bleeding_utility=res["spontaneous_bleeding_utility"][i],
            joint_bleeding_utility=res["joint_bleeding_utility"][i],
            life_threatening_bleeding_utility=res["life_threatening_bleeding_utility"][
                i
            ],
            death_utility=res["death_utility"][i],
            # Costs
            per_unit_price=res["per_unit_price"][i],
            costs_discount_rate=res["costs_discount_rate"][i],
            prophylaxis_background_factor_consumption_per_kg=res[
                "prophylaxis_background_factor_consumption_per_kg"
            ][i],
            factor_consumption_per_spontaneous_bleeding_per_kg=res[
                "factor_consumption_per_spontaneous_bleeding_per_kg"
            ][i],
            factor_consumption_per_joint_bleeding_per_kg=res[
                "factor_consumption_per_joint_bleeding_per_kg"
            ][i],
            factor_consumption_per_life_threatening_bleeding_per_kg=res[
                "factor_consumption_per_life_threatening_bleeding_per_kg"
            ][i],
        )
