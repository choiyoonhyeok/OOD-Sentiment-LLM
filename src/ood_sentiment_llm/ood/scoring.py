from __future__ import annotations

import numpy as np


# ============================================================
# 1) MSP (Maximum Softmax Probability)
# ============================================================

def score_msp(probs: np.ndarray) -> np.ndarray:
    """
    MSP 기반 OOD 점수.

    Parameters
    ----------
    probs : np.ndarray, shape (N, C)
        소프트맥스 확률값. 각 행 합은 1이라고 가정.

    Returns
    -------
    scores : np.ndarray, shape (N,)
        1 - max_prob.
        값이 클수록(= 최대 클래스 확률이 작을수록) OOD일 가능성이 크다고 해석.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"`probs` must be 2D (N, C), got shape {probs.shape}")

    max_prob = probs.max(axis=1)
    scores = 1.0 - max_prob
    return scores


# ============================================================
# 2) Energy 기반 점수 (probs / logits)
# ============================================================

def score_energy_from_probs(probs: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    softmax 확률에서 energy score 계산.

    논문에서의 에너지는 보통 logit 기반이지만,
    이 프로젝트에서는 확률값만 있는 경우를 위해 아래와 같이 정의:

        scaled = probs / T
        scores = T * log(sum(exp(scaled), axis=1))

    Parameters
    ----------
    probs : np.ndarray, shape (N, C)
    T : float
        temperature. T>0.

    Returns
    -------
    scores : np.ndarray, shape (N,)
        값이 클수록 OOD 경향이 큰 score.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"`probs` must be 2D (N, C), got shape {probs.shape}")
    if T <= 0:
        raise ValueError("`T` must be positive.")

    scaled = probs / T
    # overflow 방지를 위해 exp 전에 클리핑 (선택 사항)
    scaled = np.clip(scaled, -50.0, 50.0)

    exp_scaled = np.exp(scaled)
    sum_exp = exp_scaled.sum(axis=1)
    scores = T * np.log(sum_exp + 1e-12)
    return scores


def score_energy_from_logits(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """
    logit에서 Energy-based OOD score 계산.

    Liu et al. (2020) Energy-based OOD Detection의 정의:
        E(x) = -T * log( sum_c exp(f_c(x) / T) )

    여기서는 DMResult와 일관되게
    "점수가 클수록 OOD" 되도록 -E(x)를 반환.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)
    T : float

    Returns
    -------
    scores : np.ndarray, shape (N,)
        -E(x). 값이 클수록 OOD.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"`logits` must be 2D (N, C), got shape {logits.shape}")
    if T <= 0:
        raise ValueError("`T` must be positive.")

    scaled = logits / T
    # log-sum-exp 안정화
    m = scaled.max(axis=1, keepdims=True)
    sum_exp = np.exp(scaled - m).sum(axis=1)
    log_sum_exp = np.log(sum_exp + 1e-12) + m.squeeze(1)

    energy = -T * log_sum_exp         # E(x)
    scores = -energy                  # -E(x): 클수록 OOD
    return scores


# ============================================================
# 3) Mahalanobis Distance 기반 점수
# ============================================================

def fit_md(feats_ind: np.ndarray, reg_eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """
    IND 임베딩으로 Mahalanobis distance용 통계량(mu, inv_cov) 추정.

    Parameters
    ----------
    feats_ind : np.ndarray, shape (N, D)
        IND(인-디스트리뷰션) 임베딩들.
    reg_eps : float
        공분산 대각 성분에 추가할 작은 값 (정규화/역행렬 안정화용).

    Returns
    -------
    mu : np.ndarray, shape (D,)
        평균 벡터.
    inv_cov : np.ndarray, shape (D, D)
        공분산 행렬의 역행렬.
    """
    X = np.asarray(feats_ind, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"`feats_ind` must be 2D (N, D), got shape {X.shape}")

    mu = X.mean(axis=0)
    # rowvar=False → 각 열이 변수(D), 행이 샘플(N)
    cov = np.cov(X, rowvar=False)

    # 수치 안정화를 위해 대각에 reg_eps 추가
    cov = cov + reg_eps * np.eye(cov.shape[0], dtype=cov.dtype)

    inv_cov = np.linalg.inv(cov)
    return mu, inv_cov


def score_md(
    feats: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
) -> np.ndarray:
    """
    Mahalanobis distance를 이용한 OOD score.

    Parameters
    ----------
    feats : np.ndarray, shape (N, D)
    mu : np.ndarray, shape (D,)
    inv_cov : np.ndarray, shape (D, D)

    Returns
    -------
    scores : np.ndarray, shape (N,)
        d(x) = sqrt((x - mu)^T Σ^{-1} (x - mu)).
        값이 클수록 IND 평균에서 멀리 떨어져 있으므로 OOD 경향이 크다고 해석.
    """
    X = np.asarray(feats, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    inv_cov = np.asarray(inv_cov, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"`feats` must be 2D (N, D), got shape {X.shape}")
    if mu.ndim != 1:
        raise ValueError(f"`mu` must be 1D (D,), got shape {mu.shape}")
    if inv_cov.ndim != 2:
        raise ValueError(f"`inv_cov` must be 2D (D, D), got shape {inv_cov.shape}")

    diff = X - mu
    left = diff @ inv_cov
    d_squared = np.sum(left * diff, axis=1)
    d_squared = np.maximum(d_squared, 0.0)   # 음수 방지 (수치 오차 보정)
    scores = np.sqrt(d_squared)
    return scores
