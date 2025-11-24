# src/ood_sentiment_llm/pipeline/full_pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

from ..config import ARTIFACT_DIR
from ..utils import ensure_dir
from ..ood import (
    score_msp,
    score_energy_from_probs,
    fit_md,
    score_md,
    DMResult,
)
from ..embedding import run_openai_embedding
from ..models.pn_train import main as train_pn_main
from ..models.pn_eval import (
    main as eval_pn_main,
    load_pn_checkpoint,
    forward_pn,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# 헬퍼: 아티팩트 경로 / 데이터 로드
# ---------------------------------------------------------

def get_artifact_dir() -> Path:
    """
    config.ARTIFACT_DIR(<root>/artifacts)를 기준으로 디렉터리 보장.
    """
    ensure_dir(ARTIFACT_DIR)
    return ARTIFACT_DIR


def load_ind_ood_embeddings(
    artifact_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OOD 평가용 IND / OOD 임베딩 로드.

    우선순위:
    1) embeddings_IND_X.npy / embeddings_OOD_X.npy
    2) embeddings_test_X.npy / embeddings_ood_test_X.npy
    """
    if artifact_dir is None:
        artifact_dir = get_artifact_dir()

    ind_path_1 = artifact_dir / "embeddings_IND_X.npy"
    ood_path_1 = artifact_dir / "embeddings_OOD_X.npy"

    if ind_path_1.exists() and ood_path_1.exists():
        X_IND = np.load(ind_path_1)
        X_OOD = np.load(ood_path_1)
        return X_IND, X_OOD

    ind_path_2 = artifact_dir / "embeddings_test_X.npy"
    ood_path_2 = artifact_dir / "embeddings_ood_test_X.npy"

    if ind_path_2.exists() and ood_path_2.exists():
        X_IND = np.load(ind_path_2)
        X_OOD = np.load(ood_path_2)
        return X_IND, X_OOD

    raise FileNotFoundError(
        "IND/OOD 임베딩 파일을 찾을 수 없습니다.\n"
        f" - {ind_path_1}\n"
        f" - {ood_path_1}\n"
        f" - {ind_path_2}\n"
        f" - {ood_path_2}\n"
    )


def load_test_labels(
    artifact_dir: Path | None = None,
) -> np.ndarray:
    """
    분류용 테스트 라벨 로드.
    """
    if artifact_dir is None:
        artifact_dir = get_artifact_dir()

    y_path = artifact_dir / "embeddings_test_y.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"테스트 라벨 파일을 찾을 수 없습니다: {y_path}")
    return np.load(y_path)


# ---------------------------------------------------------
# 1. 단계별 실행 러너
# ---------------------------------------------------------

def run_embedding() -> None:
    """
    OpenAI Embedding을 이용해 train/test/OOD 임베딩 생성.
    - artifacts/ 밑에 embeddings_*.npy 저장
    """
    run_openai_embedding()


def run_train_pn() -> None:
    """
    PN classifier 학습.
    - 입력: embeddings_train_X.npy / embeddings_train_y.npy
    - 출력: best_pn_classifier.pt
    """
    train_pn_main()


def run_eval_pn() -> None:
    """
    PN classifier 테스트 평가.
    - 입력: best_pn_classifier.pt, embeddings_test_X.npy / embeddings_test_y.npy
    """
    eval_pn_main()


# ---------------------------------------------------------
# 2. OOD 평가
# ---------------------------------------------------------

def run_ood_eval_from_artifacts() -> Dict[str, DMResult]:
    """
    artifacts 디렉터리의 임베딩과 PN 모델을 사용해
    MSP / Energy / MD 기반 OOD 성능을 평가.

    Returns
    -------
    results : Dict[str, DMResult]
        key: "msp", "energy", "md"
    """
    artifact_dir = get_artifact_dir()

    # 1) IND / OOD 임베딩 로드
    X_IND, X_OOD = load_ind_ood_embeddings(artifact_dir)
    print(f"[OOD] X_IND: {X_IND.shape}, X_OOD: {X_OOD.shape}")

    # 2) PN 모델 로드
    in_dim = X_IND.shape[1]
    model_proj, model_clf, _, _, _ = load_pn_checkpoint(
        artifact_dir=artifact_dir,
        in_dim=in_dim,
        device=DEVICE,
    )

    # 3) PN 출력 확률(probs) 계산 (IND / OOD 모두)
    probs_IND, _ = forward_pn(
        model_proj=model_proj,
        model_clf=model_clf,
        X=X_IND,
        batch_size=512,
        device=DEVICE,
    )
    probs_OOD, _ = forward_pn(
        model_proj=model_proj,
        model_clf=model_clf,
        X=X_OOD,
        batch_size=512,
        device=DEVICE,
    )

    probs_all = np.vstack([probs_IND, probs_OOD])
    y_ood = np.concatenate(
        [
            np.zeros(len(X_IND), dtype=np.int64),  # IND = 0
            np.ones(len(X_OOD), dtype=np.int64),   # OOD = 1
        ]
    )

    print("[OOD] probs_all:", probs_all.shape, "y_ood:", y_ood.shape)

    # 4) OOD Score 계산
    # MSP (클수록 OOD)
    msp_scores = score_msp(probs_all)

    # Energy (probs 기반, 클수록 OOD)
    energy_scores = score_energy_from_probs(probs_all, T=1.0)

    # Mahalanobis Distance: 여기서는 원본 임베딩 공간(X)을 사용
    mu, inv_cov = fit_md(X_IND)
    md_scores = score_md(np.vstack([X_IND, X_OOD]), mu=mu, inv_cov=inv_cov)

    # 5) DMResult 계산
    def _eval(scores: np.ndarray) -> DMResult:
        dm = DMResult()
        dm.compute(y_true=y_ood, scores=scores)
        return dm

    results: Dict[str, DMResult] = {
        "msp": _eval(msp_scores),
        "energy": _eval(energy_scores),
        "md": _eval(md_scores),
    }

    print("\n[OOD] Score summary:")
    for name, dm in results.items():
        print(f"--- {name.upper()} ---")
        dm.summary()

    return results


# ---------------------------------------------------------
# 3. 전체 파이프라인
# ---------------------------------------------------------

def run_full_pipeline(
    do_embedding: bool = True,
    do_train: bool = True,
    do_eval: bool = True,
    do_ood_eval: bool = True,
) -> Dict[str, Any]:
    """
    한 번에 전체 파이프라인 실행.

    Parameters
    ----------
    do_embedding : bool
        True 이면 OpenAI 임베딩 생성부터 수행.
    do_train : bool
        True 이면 PN classifier 학습.
    do_eval : bool
        True 이면 PN 테스트 분류 평가.
    do_ood_eval : bool
        True 이면 MSP/Energy/MD OOD 평가.

    Returns
    -------
    info : Dict[str, Any]
        실행 단계별 간단한 결과 요약.
    """
    info: Dict[str, Any] = {}

    if do_embedding:
        print("[Pipeline] Step 1: Embedding 생성")
        run_embedding()
        info["embedding"] = "done"
    else:
        info["embedding"] = "skipped"

    if do_train:
        print("[Pipeline] Step 2: PN classifier 학습")
        run_train_pn()
        info["train_pn"] = "done"
    else:
        info["train_pn"] = "skipped"

    if do_eval:
        print("[Pipeline] Step 3: PN classifier 평가")
        run_eval_pn()
        info["eval_pn"] = "done"
    else:
        info["eval_pn"] = "skipped"

    if do_ood_eval:
        print("[Pipeline] Step 4: OOD 평가")
        results = run_ood_eval_from_artifacts()
        info["ood_results"] = {k: v.to_dict() for k, v in results.items()}
    else:
        info["ood_results"] = None

    return info


if __name__ == "__main__":
    run_full_pipeline()
