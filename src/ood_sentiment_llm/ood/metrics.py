from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


@dataclass
class DMResult:
    """
    Distance-based OOD Metrics Container

    가정
    ----
    * y_true: 0 = IND, 1 = OOD
    * scores: 값이 클수록 OOD에 더 가깝다고 가정
    * ROC 기준: score >= threshold → predicted OOD
    """
    auroc: float = np.nan
    aupr: float = np.nan
    fpr95: float = np.nan

    # ROC 곡선 전체도 같이 보관
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None

    # 원본 데이터 (옵션)
    y_true: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None

    def __call__(self, y_true, scores) -> "DMResult":
        """DMResult()(y_true, scores) 형태로 바로 사용 가능하게."""
        return self.compute(y_true, scores)

    # --------------------------------------------------------
    # 핵심 계산
    # --------------------------------------------------------
    def compute(self, y_true, scores) -> "DMResult":
        y = np.asarray(y_true, dtype=np.int32)
        s = np.asarray(scores, dtype=np.float64)

        if y.shape[0] != s.shape[0]:
            raise ValueError(f"len(y_true)={len(y)} != len(scores)={len(s)}")

        # 저장
        self.y_true = y
        self.scores = s

        # AUROC
        self.auroc = float(roc_auc_score(y, s))

        # ROC 곡선 (fpr, tpr, thresholds)
        fpr, tpr, thr = roc_curve(y, s)
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thr

        # FPR at 95% TPR
        target_tpr = 0.95
        idx = np.where(tpr >= target_tpr)[0]
        self.fpr95 = float(fpr[idx[0]]) if len(idx) > 0 else 1.0

        # AUPR (Positive=OOD=1 기준)
        self.aupr = float(average_precision_score(y, s))

        return self

    # --------------------------------------------------------
    # 출력/요약
    # --------------------------------------------------------
    def summary(self) -> None:
        print(f"AUROC : {self.auroc:.4f}")
        print(f"AUPR  : {self.aupr:.4f}")
        print(f"FPR95 : {self.fpr95:.4f}")
        print("========================")

    def to_dict(self) -> dict:
        """결과를 dict로 반환 (로깅/CSV 저장용)."""
        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        """단일 행 DataFrame으로 변환."""
        return pd.DataFrame([self.to_dict()])

    # --------------------------------------------------------
    # 시각화 유틸
    # --------------------------------------------------------
    def plot_roc(self, ax: Optional[plt.Axes] = None, label: str = "ROC") -> plt.Axes:
        """
        ROC 곡선 플롯.
        compute()가 먼저 호출되어야 함.
        """
        if self.fpr is None or self.tpr is None:
            raise RuntimeError("ROC 곡선 정보가 없습니다. 먼저 compute()를 호출하세요.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        ax.plot(self.fpr, self.tpr, lw=2, label=f"{label} (AUROC={self.auroc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True, ls=":", alpha=0.5)
        return ax

    def plot_hist(
        self,
        ax: Optional[plt.Axes] = None,
        bins: int = 50,
        alpha: float = 0.6,
        ind_label: str = "IND",
        ood_label: str = "OOD",
    ) -> plt.Axes:
        """
        IND vs OOD 스코어 히스토그램.
        y_true, scores 가 저장되어 있어야 함.
        """
        if self.y_true is None or self.scores is None:
            raise RuntimeError("y_true/scores 정보가 없습니다. 먼저 compute()를 호출하세요.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        y = self.y_true
        s = self.scores

        ind_scores = s[y == 0]
        ood_scores = s[y == 1]

        ax.hist(ind_scores, bins=bins, alpha=alpha, label=ind_label)
        ax.hist(ood_scores, bins=bins, alpha=alpha, label=ood_label)
        ax.set_xlabel("OOD score (higher → more OOD-like)")
        ax.set_ylabel("Count")
        ax.set_title("IND vs OOD score histogram")
        ax.legend()
        return ax
