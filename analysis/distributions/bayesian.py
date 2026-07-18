from collections.abc import Sequence

import arviz as az
import numpy as np
import pymc as pm

from analysis.distributions.base import ConvergenceDiagnostics
from persistence.schemas.clinicals import StudyEstimate


class Bayesian:
    def __init__(self, studies: Sequence[StudyEstimate]) -> None:

        if len(studies) == 0:
            raise ValueError("At least one study is required.")

        self.studies = studies
        self.k = len(studies)

        self.means = np.asarray([s.mean for s in studies], dtype=float)
        self.stds = np.asarray([s.sd for s in studies], dtype=float)
        self.sizes = np.asarray([s.size for s in studies], dtype=float)

        if np.any(self.means <= 0):
            raise ValueError("All means must be positive for log-normal meta-analysis.")

        if np.any(self.stds <= 0):
            raise ValueError("All SDs must be positive.")

        if np.any(self.sizes <= 0):
            raise ValueError("All sample sizes must be positive.")

        self.ses = self.stds / np.sqrt(self.sizes)

        self.log_means = np.log(self.means)

        self.log_ses = np.clip(
            self.ses / self.means,
            1e-6,
            None,
        )

        self.within_var = self.log_ses**2

        self.model: pm.Model | None = None
        self.trace: az.InferenceData | None = None

        self._built = False
        self._sampled = False

        self._rng = np.random.default_rng()

        self._mcmc_config = {
            "draws": 2500,
            "tune": 2000,
            "chains": 4,
            "cores": 4,
            "target_accept": 0.95,
            "random_seed": 42,
        }

    def configure_mcmc(self, **kwargs) -> None:
        self._mcmc_config.update(kwargs)

    def _ensure_fitted(self) -> None:
        if not self._sampled:
            self.fit(**self._mcmc_config)

    def build_model(
        self,
        mu_prior: dict[str, float] | None = None,
        tau_prior: dict[str, float] | None = None,
    ) -> pm.Model:

        if mu_prior is None:
            mu_prior = {
                "mu": float(np.mean(self.log_means)),
                "sigma": 2.0,
            }

        if tau_prior is None:
            tau_prior = {
                "sigma": 0.5,
            }

        self.model = pm.Model()

        with self.model:

            mu = pm.Normal(
                "mu",
                mu=mu_prior["mu"],
                sigma=mu_prior["sigma"],
            )

            tau = pm.HalfNormal(
                "tau",
                sigma=tau_prior["sigma"],
            )

            z = pm.Normal(
                "z",
                mu=0,
                sigma=1,
                shape=self.k,
            )

            theta = pm.Deterministic(
                "theta",
                mu + z * tau,
            )

            pm.Normal(
                "y_obs",
                mu=theta,
                sigma=self.log_ses,
                observed=self.log_means,
            )

            pm.Deterministic(
                "effect_size",
                pm.math.exp(mu),
            )

        self._built = True

        return self.model

    def fit(
        self,
        draws: int = 2500,
        tune: int = 2000,
        chains: int = 4,
        cores: int = 4,
        target_accept: float = 0.95,
        random_seed: int = 42,
        mu_prior: dict[str, float] | None = None,
        tau_prior: dict[str, float] | None = None,
        **kwargs,
    ) -> az.InferenceData:

        if not self._built:
            self.build_model(
                mu_prior=mu_prior,
                tau_prior=tau_prior,
            )

        self._rng = np.random.default_rng(random_seed)

        with self.model:

            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
                progressbar=False,
                **kwargs,
            )

        self._sampled = True

        return self.trace

    def sample(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:

        self._ensure_fitted()

        if rng is None:
            rng = self._rng

        mu_post = self.trace.posterior["mu"].values.reshape(-1)
        tau_post = self.trace.posterior["tau"].values.reshape(-1)

        idx = rng.choice(
            len(mu_post),
            size=n,
            replace=True,
        )

        tau = np.clip(
            tau_post[idx],
            1e-6,
            5.0,
        )

        theta = rng.normal(
            loc=mu_post[idx],
            scale=tau,
        )

        y = np.exp(theta)

        y = np.clip(
            y,
            1e-6,
            np.percentile(y, 99.5),
        )

        return y

    def sample_study_level(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:

        self._ensure_fitted()

        if rng is None:
            rng = self._rng

        theta = self.trace.posterior["theta"].values

        flat = theta.reshape(-1, self.k)

        idx = rng.choice(
            len(flat),
            size=n,
            replace=True,
        )

        return np.exp(flat[idx])

    def point(self) -> float:

        self._ensure_fitted()

        mu = self.trace.posterior["mu"].values.mean()

        return float(np.exp(mu))

    def get_distribution_stats(self) -> dict[str, float]:

        self._ensure_fitted()

        mu_samples = np.exp(self.trace.posterior["mu"].values.flatten())

        tau_samples = self.trace.posterior["tau"].values.flatten()

        q = np.quantile(
            mu_samples,
            [0.025, 0.25, 0.5, 0.75, 0.975],
        )

        typical_within = float(np.median(self.within_var))

        i2 = tau_samples**2 / (tau_samples**2 + typical_within)

        return {
            "mu_mean": float(np.mean(mu_samples)),
            "mu_std": float(np.std(mu_samples)),
            "mu_median": float(np.median(mu_samples)),
            "mu_95ci_lower": float(q[0]),
            "mu_95ci_upper": float(q[-1]),
            "tau_mean": float(np.mean(tau_samples)),
            "tau_median": float(np.median(tau_samples)),
            "I2_mean": float(np.mean(i2)),
            "I2_median": float(np.median(i2)),
            "point_estimate": self.point(),
        }

    def summary(
        self,
        var_names: list | None = None,
        **kwargs,
    ):

        self._ensure_fitted()

        if var_names is None:
            var_names = [
                "effect_size",
                "tau",
            ]

        return az.summary(
            self.trace,
            var_names=var_names,
            **kwargs,
        )

    def convergence_diagnostics(self) -> ConvergenceDiagnostics:

        self._ensure_fitted()

        rhat_ds = az.rhat(self.trace)
        ess_ds = az.ess(self.trace)

        rhat_max = float(rhat_ds.to_array().max())
        ess_min = int(ess_ds.to_array().min())

        divergences = int(
            getattr(
                self.trace.sample_stats,
                "diverging",
                0,
            )
            .sum()
            .item()
        )

        converged = bool((rhat_max < 1.1) and (divergences == 0))

        return {
            "r_hat": rhat_max,
            "ess": ess_min,
            "divergences": divergences,
            "converged": converged,
        }
