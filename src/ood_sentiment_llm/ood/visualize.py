from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


def l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    벡터 a, b 사이의 L2 거리.

    a : (N, D)
    b : (D,)
    반환 : (N,)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b, axis=1)


# ============================================================
# 1) PN vs OOD 프로토타입 거리 히스토그램
# ============================================================

def plot_prototype_distance_hist(
    pn_X: np.ndarray,
    ood_X: np.ndarray,
    proto_neg: np.ndarray,
    proto_pos: np.ndarray,
    save_path: Optional[Path] = None,
) -> Tuple[float, float]:
    """
    PN(IND) vs OOD 에 대해
    "가까운 프로토타입까지의 거리" 분포를 히스토그램으로 시각화.

    Parameters
    ----------
    pn_X : np.ndarray, shape (N_pn, D)
        PN(IND) 임베딩.
    ood_X : np.ndarray, shape (N_ood, D)
        OOD 임베딩.
    proto_neg, proto_pos : np.ndarray, shape (D,)
        PN 클래스별 프로토타입 벡터.
    save_path : Optional[Path]
        지정하면 png로 저장.

    Returns
    -------
    auroc : float
        "거리 score(클수록 OOD)"를 기준으로 OOD 분류 AUROC.
    ks_stat : float
        KS 검정 통계량 (분포 차이 정도).
    """
    pn_X = np.asarray(pn_X, dtype=np.float64)
    ood_X = np.asarray(ood_X, dtype=np.float64)

    dist_pn = np.minimum(
        l2_distance(pn_X, proto_neg),
        l2_distance(pn_X, proto_pos),
    )
    dist_ood = np.minimum(
        l2_distance(ood_X, proto_neg),
        l2_distance(ood_X, proto_pos),
    )

    # score = 거리 그대로 사용 (클수록 OOD)
    y_true = np.concatenate([
        np.zeros_like(dist_pn, dtype=np.int32),
        np.ones_like(dist_ood, dtype=np.int32),
    ])
    scores = np.concatenate([dist_pn, dist_ood])

    auroc = float(roc_auc_score(y_true, scores))
    ks_stat, _ = ks_2samp(dist_pn, dist_ood)

    plt.figure(figsize=(9, 5))
    plt.hist(dist_pn, bins=50, alpha=0.6, label="IND (PN, min-dist→prototype)")
    plt.hist(dist_ood, bins=50, alpha=0.6, label="OOD (min-dist→prototype)")
    plt.xlabel("Min distance to nearest prototype")
    plt.ylabel("Count")
    plt.title(f"PN vs OOD distance histogram (AUROC={auroc:.3f}, KS={ks_stat:.3f})")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot_prototype_distance_hist] 히스토그램 저장: {save_path}")

    return auroc, ks_stat


# ============================================================
# 2) IND vs OOD 임베딩 산점도(PCA 2D)
# ============================================================

def plot_ind_ood_scatter_pca(
    X_ind: np.ndarray,
    X_ood: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    IND / OOD 임베딩을 합쳐서 간단한 PCA(=SVD) 2D 시각화.

    Parameters
    ----------
    X_ind : np.ndarray, shape (N_ind, D)
    X_ood : np.ndarray, shape (N_ood, D)
    save_path : Optional[Path]
        지정 시 png로 저장.
    """
    X_ind = np.asarray(X_ind, dtype=np.float64)
    X_ood = np.asarray(X_ood, dtype=np.float64)

    X = np.vstack([X_ind, X_ood])     # (N_ind + N_ood, D)
    # 중심화
    Xc = X - X.mean(axis=0, keepdims=True)

    # SVD 기반 PCA 2D
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    Z_ind = Z[: len(X_ind)]
    Z_ood = Z[len(X_ind) :]

    plt.figure(figsize=(8, 6))
    plt.scatter(Z_ind[:, 0], Z_ind[:, 1], s=10, alpha=0.7, label="IND")
    plt.scatter(Z_ood[:, 0], Z_ood[:, 1], s=10, alpha=0.7, label="OOD")
    plt.title("IND vs OOD scatter (PCA)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        print(f"[plot_ind_ood_scatter_pca] 산점도 저장: {save_path}")
