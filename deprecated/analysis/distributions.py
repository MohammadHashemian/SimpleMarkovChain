from typing import Dict, Optional, Sequence
from pydantic import BaseModel
from persistence.schemas.clinicals import StudyEstimate
import numpy as np


class BayesianHierarchicalDistribution:
    def __init__(self, studies: Sequence[StudyEstimate]) -> None:
        self.studies = studies
        self.means = np.array([s.mean for s in studies])
        self.sds = np.array([s.sd for s in studies])
        self.log_means = np.log(self.means + 1e-9)
        self.log_sds = self.sds / (self.means + 1e-9)
        self.mu_prior_mean = np.mean(self.log_means)
        self.mu_prior_sd = 1.0
        self.tau_prior_scale = 0.75

    def _sample_mu_tau(self, n: int, rng: np.random.Generator):
        if len(self.log_means) == 0:
            return (np.full(n, self.mu_prior_mean), np.full(n, 0.3))
        weights = 1.0 / (self.log_sds**2 + 1e-8)
        sum_w = np.sum(weights)
        mu_hat = np.sum(weights * self.log_means) / sum_w
        q = np.sum(weights * (self.log_means - mu_hat) ** 2)
        df = len(self.log_means) - 1
        c = sum_w - np.sum(weights**2) / sum_w
        tau2_dl = max(0.0, (q - df) / (c + 1e-8))
        tau_hat = np.sqrt(tau2_dl)
        var_mu = 1.0 / sum_w + tau_hat**2
        mu_sd = np.sqrt(var_mu) * 1.25
        mu = rng.normal(mu_hat, mu_sd, size=n)
        if len(self.studies) <= 4:
            mu = 0.8 * mu + 0.2 * self.mu_prior_mean
        tau_log_mean = np.log(max(tau_hat, 0.12))
        tau = rng.lognormal(mean=tau_log_mean, sigma=0.45, size=n)
        tau = np.clip(tau, 0.08, 0.95)
        return mu, tau

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mu, tau = self._sample_mu_tau(n, rng)
        theta = rng.normal(mu, tau)
        sigma_typical = np.median(self.log_sds)
        log_y = rng.normal(theta, sigma_typical)
        y = np.exp(log_y)
        y = np.clip(y, 0.1, np.percentile(y, 99.0))
        return y

    def point(self) -> float:
        return float(np.exp(self.mu_prior_mean))


class CohortBayesianHierarchicalDistribution:
    def __init__(self, studies: Sequence[StudyEstimate]) -> None:
        self.studies = studies
        means = np.asarray([s.mean for s in studies], dtype=float)
        sds = np.asarray([s.sd for s in studies], dtype=float)
        means = np.clip(means, 1e-6, None)
        sds = np.clip(sds, 1e-6, None)
        self.log_sds = np.sqrt(np.log(1.0 + (sds**2 / means**2)))
        self.log_means = np.log(means) - 0.5 * self.log_sds**2
        weights = 1.0 / (self.log_sds**2)
        self.mu_hat = np.sum(weights * self.log_means) / np.sum(weights)
        q = np.sum(weights * (self.log_means - self.mu_hat) ** 2)
        df = len(self.log_means) - 1
        c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau2 = max(0.0, (q - df) / max(c, 1e-9))
        self.tau_hat = np.sqrt(tau2)
        self.tau_hat = min(self.tau_hat, 0.75)

    def sample(self, n: int, rng: np.random.Generator, patient_cv: float = 0.35) -> np.ndarray:
        mu = rng.normal(loc=self.mu_hat, scale=max(self.tau_hat, 0.05))
        sigma_patient = np.sqrt(np.log(1.0 + patient_cv**2))
        patient_log_rates = rng.normal(loc=mu, scale=sigma_patient, size=n)
        y = np.exp(patient_log_rates)
        y = np.clip(y, 0.01, 80.0)
        return y

    def point(self) -> float:
        return float(np.exp(self.mu_hat))


class ABRDistribution(BaseModel):
    study_data: list = []

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        num_studies = len(self.study_data)
        shapes = []
        scales = []
        eps = 1e-8
        for study in self.study_data:
            mu, sigma = study.mean, study.sd
            if mu <= 0 or sigma <= 0:
                shapes.append(None)
                scales.append(None)
                continue
            k = max((mu / (sigma + eps)) ** 2, eps)
            theta = max((sigma**2) / (mu + eps), eps)
            shapes.append(k)
            scales.append(theta)
        study_samples = np.zeros((n, num_studies))
        for j in range(num_studies):
            if shapes[j] is None:
                study_samples[:, j] = max(0, self.study_data[j].mean)
            else:
                from scipy.stats import gamma as gamma_dist
                study_samples[:, j] = gamma_dist.rvs(a=shapes[j], scale=scales[j], size=n, random_state=rng)
        weights = rng.dirichlet(alpha=np.ones(num_studies), size=n)
        return np.sum(weights * study_samples, axis=1)
