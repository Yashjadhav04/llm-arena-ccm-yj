from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class PairwisePreferenceResult:
    model_scores: pd.DataFrame
    coefficients: pd.DataFrame
    metrics: dict[str, float]


def fit_pairwise_logit(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    l2_penalty: float = 1e-4,
    maxiter: int = 300,
) -> PairwisePreferenceResult:
    feature_columns = feature_columns or []
    required = ["model_a", "model_b", "winner_binary", *feature_columns]
    data = frame[required].dropna().copy()

    if data.empty:
        raise ValueError("No rows available to fit the pairwise model.")

    models = sorted(set(data["model_a"]).union(set(data["model_b"])))
    if len(models) < 2:
        raise ValueError("Need at least two unique models to fit the pairwise model.")

    reference_model = models[-1]
    free_models = models[:-1]
    free_index = {name: idx for idx, name in enumerate(free_models)}
    full_index = {name: idx for idx, name in enumerate(models)}

    a_ref = data["model_a"].map(lambda name: free_index.get(name, -1)).to_numpy()
    b_ref = data["model_b"].map(lambda name: free_index.get(name, -1)).to_numpy()
    y = data["winner_binary"].to_numpy(dtype=float)
    x = (
        data[feature_columns].to_numpy(dtype=float)
        if feature_columns
        else np.zeros((len(data), 0), dtype=float)
    )

    n_free = len(free_models)
    n_beta = x.shape[1]

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        skills_free = theta[:n_free]
        betas = theta[n_free:]
        skills = np.zeros(len(models), dtype=float)
        skills[:-1] = skills_free
        return skills, betas

    def linear_term(skills: np.ndarray, betas: np.ndarray) -> np.ndarray:
        eta = np.zeros(len(data), dtype=float)
        a_mask = a_ref >= 0
        b_mask = b_ref >= 0
        eta[a_mask] += skills[a_ref[a_mask]]
        eta[b_mask] -= skills[b_ref[b_mask]]
        if n_beta:
            eta += x @ betas
        return eta

    def objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        skills, betas = unpack(theta)
        eta = linear_term(skills, betas)
        p = expit(eta)
        eps = 1e-9
        loglike = y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
        penalty = 0.5 * l2_penalty * float(np.dot(theta, theta))
        loss = float(-np.sum(loglike) + penalty)

        residual = p - y
        grad_skills = np.zeros(n_free, dtype=float)
        a_mask = a_ref >= 0
        b_mask = b_ref >= 0
        np.add.at(grad_skills, a_ref[a_mask], residual[a_mask])
        np.add.at(grad_skills, b_ref[b_mask], -residual[b_mask])
        grad_betas = x.T @ residual if n_beta else np.zeros(0, dtype=float)

        grad = np.concatenate([grad_skills, grad_betas]) + l2_penalty * theta
        return loss, grad

    theta0 = np.zeros(n_free + n_beta, dtype=float)
    result = minimize(
        objective_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter},
    )
    if not result.success:
        raise RuntimeError(f"Pairwise optimization failed: {result.message}")

    skills, betas = unpack(result.x)
    eta = linear_term(skills, betas)
    probs = expit(eta)
    preds = (probs >= 0.5).astype(float)

    centered_skills = skills - skills.mean()
    scores = pd.DataFrame(
        {
            "model": models,
            "score": centered_skills,
        }
    ).sort_values("score", ascending=False, ignore_index=True)

    coeff_names = feature_columns.copy()
    coefficients = pd.DataFrame(
        {
            "term": coeff_names,
            "estimate": betas,
        }
    )

    eps = 1e-9
    log_loss = float(-np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))
    accuracy = float(np.mean(preds == y))
    base_rate = float(y.mean())
    null_probs = np.full(len(y), base_rate, dtype=float)
    null_log_loss = float(
        -np.mean(y * np.log(null_probs + eps) + (1 - y) * np.log(1 - null_probs + eps))
    )

    metrics = {
        "n_rows": float(len(data)),
        "n_models": float(len(models)),
        "reference_model": reference_model,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "null_log_loss": null_log_loss,
        "mean_model_a_win_rate": base_rate,
    }

    return PairwisePreferenceResult(
        model_scores=scores,
        coefficients=coefficients,
        metrics=metrics,
    )
