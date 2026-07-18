from collections.abc import Sequence
from typing import Any, Protocol

import arviz as az
import numpy as np
import pymc as pm
from pydantic import BaseModel, Field

from persistence.schemas.clinicals import StudyEstimate


class Distribution(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    # stochastic
    def point(self) -> float: ...


class Constant:
    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.value)

    def point(self) -> float:
        return self.value


class GammaFromMeanSD(BaseModel):
    mean: float
    sd: float

    def sample(self, n: int, rng: np.random.Generator):
        k = (self.mean / self.sd) ** 2
        theta = (self.sd**2) / self.mean
        return rng.gamma(shape=k, scale=theta, size=n)

    def point(self) -> float:
        return self.mean


class GammaFromMeanCV(BaseModel):
    mean: float
    cv: float

    def sample(self, n: int, rng: np.random.Generator):
        if self.cv <= 0:
            raise ValueError("CV must be > 0")

        shape = 1 / (self.cv**2)
        scale = self.mean * (self.cv**2)

        return rng.gamma(shape=shape, scale=scale, size=n)

    def point(self) -> float:
        return self.mean


class BetaFromMeanSD(BaseModel):
    mean: float = Field(..., gt=0, lt=1)
    sd: float | None = Field(default=None, gt=0)
    cv: float | None = Field(default=None, ge=0)

    def _resolve_sd(self) -> float:
        """
        Resolve SD from either sd or cv.
        Priority: sd > cv
        """
        if self.sd is not None:
            return self.sd

        if self.cv is not None:
            return self.mean * self.cv

        raise ValueError("Either sd or cv must be provided.")

    def _to_beta_params(self):
        sd = self._resolve_sd()
        var = sd**2

        # Feasibility check for Beta distribution
        max_var = self.mean * (1 - self.mean)
        if var >= max_var:
            raise ValueError(
                f"Invalid Beta parameters: mean={self.mean}, sd={sd}. "
                f"Variance must be < mean*(1-mean)={max_var:.6f}"
            )

        common = (self.mean * (1 - self.mean) / var) - 1

        alpha = self.mean * common
        beta = (1 - self.mean) * common

        # Numerical safety
        alpha = max(alpha, 1e-8)
        beta = max(beta, 1e-8)

        return alpha, beta

    def sample(self, n: int, rng: np.random.Generator):
        alpha, beta = self._to_beta_params()
        return rng.beta(alpha, beta, size=n)

    def point(self) -> float:
        return self.mean


# class BetaFromMeanSD(BaseModel):
#     mean: float
#     sd: float

#     def sample(self, n: int, rng: np.random.Generator):
#         var = self.sd**2
#         common = (self.mean * (1 - self.mean) / var) - 1
#         alpha = self.mean * common
#         beta = (1 - self.mean) * common
#         return rng.beta(alpha, beta, size=n)

#     def point(self) -> float:
#         return self.mean


class TriangularDist(BaseModel):
    left: float
    mode: float
    right: float

    def sample(self, n: int, rng: np.random.Generator):
        return rng.triangular(self.left, self.mode, self.right, size=n)

    def point(self) -> float:
        return (self.left + self.mode + self.right) / 3


class MixtureOfStudies:
    def __init__(self, components: Sequence[Any]) -> None:
        self.components = components

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        k = len(self.components)

        samples = np.column_stack([comp.sample(n, rng) for comp in self.components])

        weights = rng.dirichlet(np.ones(k), size=n)

        return np.sum(weights * samples, axis=1)

    def point(self) -> float:
        # simple average of component means
        return float(np.mean([c.point() for c in self.components]))


class DirichletMixture:
    """
    Model averaging via Dirichlet-weighted mixture of distributions.
    Useful when combining multiple studies/models without assuming a single true effect.
    """

    def __init__(
        self,
        components: Sequence,
        alpha: float | np.ndarray | None = None,
    ) -> None:
        self.components = components
        self.k = len(components)

        if alpha is None:
            self.alpha = np.ones(self.k)
        elif isinstance(alpha, float):
            self.alpha = np.full(self.k, alpha)
        else:
            self.alpha = np.asarray(alpha)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        component_samples = np.column_stack([c.sample(n, rng) for c in self.components])

        weights = rng.dirichlet(self.alpha, size=n)

        return np.sum(weights * component_samples, axis=1)

    def point(self) -> float:
        w = self.alpha / np.sum(self.alpha)
        return float(sum(wi * c.point() for wi, c in zip(w, self.components)))


class Bayesian:
    """
    Bayesian random-effects meta-analysis for positive/skewed outcomes

    Features
    --------
    - Log-normal hierarchical model
    - Non-centered parameterization
    - Positive-only posterior predictive samples
    - Lazy fitting
    - Stable heterogeneity prior
    - Faster and more robust than centered normal model

    Statistical model
    -----------------
    log(y_i) ~ Normal(theta_i, sigma_i)

    theta_i = mu + z_i * tau

    z_i ~ Normal(0,1)

    where:
        mu   = pooled log-effect
        tau  = between-study heterogeneity
        y_i  = observed study mean
    """

    def __init__(self, studies: Sequence[StudyEstimate]) -> None:

        if len(studies) == 0:
            raise ValueError("At least one study is required.")

        self.studies = studies
        self.k = len(studies)

        # Raw study data

        self.means = np.asarray([s.mean for s in studies], dtype=float)
        self.stds = np.asarray([s.sd for s in studies], dtype=float)
        self.sizes = np.asarray([s.size for s in studies], dtype=float)

        # Validation

        if np.any(self.means <= 0):
            raise ValueError("All means must be positive for log-normal meta-analysis.")

        if np.any(self.stds <= 0):
            raise ValueError("All SDs must be positive.")

        if np.any(self.sizes <= 0):
            raise ValueError("All sample sizes must be positive.")

        # Standard errors

        self.ses = self.stds / np.sqrt(self.sizes)

        # Log-scale transformation
        # Delta-method approximation

        self.log_means = np.log(self.means)

        self.log_ses = np.clip(
            self.ses / self.means,
            1e-6,
            None,
        )

        self.within_var = self.log_ses**2

        # Internal state

        self.model: pm.Model | None = None
        self.trace: az.InferenceData | None = None

        self._built = False
        self._sampled = False

        self._rng = np.random.default_rng()

        # Default MCMC config

        self._mcmc_config = {
            "draws": 1000,
            "tune": 1000,
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

            # Population mean (log scale)

            mu = pm.Normal(
                "mu",
                mu=mu_prior["mu"],
                sigma=mu_prior["sigma"],
            )

            # Between-study heterogeneity

            tau = pm.HalfNormal(
                "tau",
                sigma=tau_prior["sigma"],
            )
            # OR
            # tau = pm.Exponential("tau", 2)

            # Non-centered parameterization

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

            # Observation model

            pm.Normal(
                "y_obs",
                mu=theta,
                sigma=self.log_ses,
                observed=self.log_means,
            )

            # Back-transformed pooled effect

            pm.Deterministic(
                "effect_size",
                pm.math.exp(mu),
            )

        self._built = True

        return self.model

    # Fitting

    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        cores: int = 2,
        target_accept: float = 0.90,
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

        with self.model:  # type: ignore

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
        """
        Sample from posterior predictive distribution.

        Returns strictly positive values.
        """

        self._ensure_fitted()

        if rng is None:
            rng = self._rng

        mu_post = self.trace.posterior["mu"].values.reshape(-1)  # type: ignore
        tau_post = self.trace.posterior["tau"].values.reshape(-1)  # type: ignore

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

        # Optional stability clipping for PSA
        y = np.clip(
            y,
            1e-6,
            np.percentile(y, 99.5),
        )

        return y

    # Study-level posterior samples

    def sample_study_level(
        self,
        n: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:

        self._ensure_fitted()

        if rng is None:
            rng = self._rng

        theta = self.trace.posterior["theta"].values  # type: ignore

        flat = theta.reshape(-1, self.k)

        idx = rng.choice(
            len(flat),
            size=n,
            replace=True,
        )

        return np.exp(flat[idx])

    # Point estimate

    def point(self) -> float:

        self._ensure_fitted()

        mu = self.trace.posterior["mu"].values.mean()  # type: ignore

        return float(np.exp(mu))

    # Summary statistics

    def get_distribution_stats(self) -> dict[str, float]:

        self._ensure_fitted()

        mu_samples = np.exp(self.trace.posterior["mu"].values.flatten())  # type: ignore

        tau_samples = self.trace.posterior["tau"].values.flatten()  # type: ignore

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

    # ArviZ summary

    def summary(
        self,
        var_names: list | None = None,
        hdi_prob: float = 0.95,
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
            hdi_prob=hdi_prob,
            **kwargs,
        )

    # Diagnostics

    def convergence_diagnostics(self) -> dict[str, Any]:

        self._ensure_fitted()

        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)

        divergences = int(
            getattr(
                self.trace.sample_stats,  # type: ignore
                "diverging",
                0,
            )
            .sum()  # type: ignore
            .item()
        )

        converged = bool((rhat.to_array().max() < 1.1) and (divergences == 0))  # type: ignore

        return {
            "r_hat": rhat,
            "ess": ess,
            "divergences": divergences,
            "converged": converged,
        }




        # def sample(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)

        num_studies = len(self.study_data)

        U = rng.uniform(size=(n, num_studies + 1))

        shapes = []
        scales = []

        # same logic, just using ABRStudy fields
        for study in self.study_data:
            mu = study.mean
            sigma = study.sd

            if sigma <= 0 or mu <= 0:
                shapes.append(None)
                scales.append(None)
                continue

            k = (mu / sigma) ** 2
            theta = (sigma**2) / mu

            shapes.append(k if k > 0 else None)
            scales.append(theta if theta > 0 else None)

        study_samples = np.zeros((n, num_studies))

        for j, study in enumerate(self.study_data):
            mu = study.mean

            if shapes[j] is None:
                study_samples[:, j] = max(0, mu)
            else:
                study_samples[:, j] = gamma_dist.ppf(
                    U[:, j],
                    a=shapes[j],
                    scale=scales[j],
                )

        weights_raw = U[:, -1][:, None] * np.ones((1, num_studies))
        weights = np.exp(weights_raw) / np.sum(
            np.exp(weights_raw), axis=1, keepdims=True
        )

        return np.sum(weights * study_samples, axis=1)
