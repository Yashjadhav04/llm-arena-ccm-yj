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
    feature_columns: list[str]


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
        feature_columns=feature_columns,
    )


def predict_pairwise_logit(
    frame: pd.DataFrame,
    result: PairwisePreferenceResult,
) -> pd.DataFrame:
    feature_columns = result.feature_columns
    required = ["model_a", "model_b", *feature_columns]
    data = frame[required].copy()

    score_map = result.model_scores.set_index("model")["score"]
    known_model_a = data["model_a"].isin(score_map.index)
    known_model_b = data["model_b"].isin(score_map.index)

    skill_a = data["model_a"].map(score_map).fillna(0.0).to_numpy(dtype=float)
    skill_b = data["model_b"].map(score_map).fillna(0.0).to_numpy(dtype=float)
    eta = skill_a - skill_b

    if feature_columns:
        coeff_map = result.coefficients.set_index("term")["estimate"]
        betas = coeff_map.reindex(feature_columns).fillna(0.0).to_numpy(dtype=float)
        x = data[feature_columns].to_numpy(dtype=float)
        eta += x @ betas

    probs = expit(eta)
    return pd.DataFrame(
        {
            "pred_proba": probs,
            "pred_label": (probs >= 0.5).astype(float),
            "known_model_a": known_model_a.to_numpy(),
            "known_model_b": known_model_b.to_numpy(),
            "known_pair": (known_model_a & known_model_b).to_numpy(),
        },
        index=frame.index,
    )


def evaluate_pairwise_logit(
    frame: pd.DataFrame,
    result: PairwisePreferenceResult,
    unknown_policy: str = "drop",
) -> dict[str, float]:
    if unknown_policy not in {"drop", "keep", "error"}:
        raise ValueError("unknown_policy must be one of: drop, keep, error")

    predictions = predict_pairwise_logit(frame, result)
    y = pd.to_numeric(frame["winner_binary"], errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(y)

    if unknown_policy == "drop":
        mask &= predictions["known_pair"].to_numpy(dtype=bool)
    elif unknown_policy == "error" and not predictions.loc[mask, "known_pair"].all():
        raise ValueError("Encountered evaluation rows with models not seen during fitting.")

    y_eval = y[mask]
    probs = predictions.loc[mask, "pred_proba"].to_numpy(dtype=float)
    preds = predictions.loc[mask, "pred_label"].to_numpy(dtype=float)

    if len(y_eval) == 0:
        raise ValueError("No valid rows remained for evaluation.")

    eps = 1e-9
    base_rate = float(y_eval.mean())
    null_probs = np.full(len(y_eval), base_rate, dtype=float)

    return {
        "n_rows": float(len(y_eval)),
        "accuracy": float(np.mean(preds == y_eval)),
        "log_loss": float(
            -np.mean(y_eval * np.log(probs + eps) + (1 - y_eval) * np.log(1 - probs + eps))
        ),
        "brier_score": float(np.mean((probs - y_eval) ** 2)),
        "null_log_loss": float(
            -np.mean(
                y_eval * np.log(null_probs + eps)
                + (1 - y_eval) * np.log(1 - null_probs + eps)
            )
        ),
        "mean_model_a_win_rate": base_rate,
        "covered_row_share": float(mask.mean()),
        "unknown_pair_rows": float((~predictions.loc[~np.isnan(y), "known_pair"]).sum()),
    }
