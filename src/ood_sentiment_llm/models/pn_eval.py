from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from ..config import ARTIFACT_DIR
from ..utils import ensure_dir, get_artifact_dir
from .heads import ProjectionHead, ClassifierHead


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# 0. 아티팩트 경로 찾기
# ----------------------------------------------------------

def find_artifact_dir() -> Path:
    """
    기본적으로 config.ARTIFACT_DIR(<root>/artifacts)를 사용하고,
    필요하면 예전 구조(./src/artifacts, ./artifacts)도 fallback으로 확인.
    """
    # 1) 새 구조
    cand = ARTIFACT_DIR
    if (cand / "embeddings_test_X.npy").exists():
        return cand.resolve()

    # 2) 혹시 옛날 구조가 남아있다면
    for other in [get_artifact_dir(), Path("src/artifacts"), Path("artifacts")]:
        if (other / "embeddings_test_X.npy").exists():
            return other.resolve()

    raise FileNotFoundError(
        "테스트 임베딩 파일을 찾을 수 없습니다.\n"
        "다음 중 한 폴더에 embeddings_test_X.npy / embeddings_test_y.npy 를 두세요:\n"
        f" - {ARTIFACT_DIR}\n"
        " - ./src/artifacts\n"
        " - ./artifacts"
    )


# ----------------------------------------------------------
# 1. 데이터 / 체크포인트 로드
# ----------------------------------------------------------

def load_test_embeddings(artifact_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    테스트 임베딩 로드.
    - embeddings_test_X.npy : (N, D)
    - embeddings_test_y.npy : (N,)
    """
    x_path = artifact_dir / "embeddings_test_X.npy"
    y_path = artifact_dir / "embeddings_test_y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "테스트 임베딩 파일을 찾을 수 없습니다.\n"
            f"- {x_path}\n"
            f"- {y_path}\n"
            "임베딩 생성 스크립트로 test 임베딩을 먼저 생성해 주세요."
        )

    X_test = np.load(x_path)
    y_test = np.load(y_path)

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"X_test({X_test.shape[0]}) / y_test({y_test.shape[0]}) 길이가 다릅니다."
        )

    return X_test, y_test


def load_pn_checkpoint(
    artifact_dir: Path,
    in_dim: int,
    device: str = DEVICE,
) -> Tuple[nn.Module, nn.Module, torch.Tensor | None, torch.Tensor | None, int]:
    """
    pn_train.py에서 저장한 best_pn_classifier.pt 체크포인트를 로드.:contentReference[oaicite:2]{index=2}

    ckpt 포맷:
        {
            "proj": model_proj.state_dict(),
            "clf":  model_clf.state_dict(),
            "proto_neg": np.ndarray(D,),
            "proto_pos": np.ndarray(D,),
            "proj_dim": int
        }
    """
    ckpt_path = artifact_dir / "best_pn_classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"모델 체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
            "먼저 pn_train.py를 실행하여 best_pn_classifier.pt 를 생성하세요."
        )

    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")

    proj_state = ckpt["proj"]
    clf_state = ckpt["clf"]
    proj_dim = int(ckpt.get("proj_dim", 256))

    # 모델 생성 (heads.py 구조에 맞춤):contentReference[oaicite:3]{index=3}
    model_proj = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim).to(device)
    model_clf = ClassifierHead(proj_dim=proj_dim, num_classes=2).to(device)

    model_proj.load_state_dict(proj_state)
    model_clf.load_state_dict(clf_state)

    # 프로토타입(나중에 OOD 분석 등에서 사용할 수 있으니 같이 로드)
    proto_neg = ckpt.get("proto_neg", None)
    proto_pos = ckpt.get("proto_pos", None)

    def _to_tensor(arr):
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    proto_neg_t = _to_tensor(proto_neg)
    proto_pos_t = _to_tensor(proto_pos)

    model_proj.eval()
    model_clf.eval()

    print(
        f"[pn_eval] Loaded PN checkpoint from {ckpt_path} "
        f"(proj_dim={proj_dim}, has_proto={proto_neg_t is not None})"
    )

    return model_proj, model_clf, proto_neg_t, proto_pos_t, proj_dim


# ----------------------------------------------------------
# 2. forward & 지표 계산
# ----------------------------------------------------------

@torch.no_grad()
def forward_pn(
    model_proj: nn.Module,
    model_clf: nn.Module,
    X: np.ndarray,
    batch_size: int = 512,
    device: str = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    고정 임베딩 X에 대해 PN classifier의 확률/예측을 계산.

    Returns
    -------
    probs : np.ndarray, shape (N, 2)
        softmax 확률.
    preds : np.ndarray, shape (N,)
        argmax 클래스 (0/1).
    """
    model_proj.eval()
    model_clf.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    N = X_tensor.shape[0]

    probs_list = []
    preds_list = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = X_tensor[start:end].to(device)  # (B, D)

        fx = model_proj(xb)               # (B, proj_dim)
        logits = model_clf(fx)            # (B, 2)
        pb = torch.softmax(logits, dim=1).cpu().numpy()
        preds_b = pb.argmax(axis=1)

        probs_list.append(pb)
        preds_list.append(preds_b)

    probs = np.vstack(probs_list)
    preds = np.concatenate(preds_list)

    return probs, preds


def evaluate_classification(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    분류 성능 지표 계산.
    - y_true, preds : 0/1 라벨
    - probs[:,1] 로 AUROC 계산 (class 1 = positive)
    """
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", pos_label=1
    )

    result = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if probs is not None:
        try:
            auroc = roc_auc_score(y_true, probs[:, 1])
            result["auroc"] = float(auroc)
        except Exception:
            result["auroc"] = float("nan")

    return result


def print_classification_report(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray | None = None,
) -> None:
    """
    scikit-learn classification_report + confusion_matrix 출력용.
    """
    print("=== Classification Report ===")
    print(classification_report(y_true, preds, digits=4))

    cm = confusion_matrix(y_true, preds)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    if probs is not None:
        try:
            auroc = roc_auc_score(y_true, probs[:, 1])
            print(f"AUROC (pos=1): {auroc:.4f}")
        except Exception as e:
            print(f"AUROC 계산 실패: {e}")


# ----------------------------------------------------------
# 3. main()
# ----------------------------------------------------------

def main() -> None:
    device = DEVICE
    print(f"[pn_eval] Using device: {device}")

    artifact_dir = find_artifact_dir()
    print(f"[pn_eval] Using artifacts at: {artifact_dir}")

    # 1) 테스트 임베딩 로드
    X_test, y_test = load_test_embeddings(artifact_dir)
    in_dim = X_test.shape[1]
    print(f"[pn_eval] Loaded TEST: X_test={X_test.shape}, y_test={y_test.shape}")

    # 2) PN 모델 + 프로토타입 로드
    model_proj, model_clf, proto_neg, proto_pos, proj_dim = load_pn_checkpoint(
        artifact_dir,
        in_dim=in_dim,
        device=device,
    )

    # 3) forward & 평가
    probs, preds = forward_pn(
        model_proj=model_proj,
        model_clf=model_clf,
        X=X_test,
        batch_size=512,
        device=device,
    )

    metrics = evaluate_classification(y_true=y_test, preds=preds, probs=probs)
    print("\n=== PN Classifier Test Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v:.4f}")

    print()
    print_classification_report(y_true=y_test, preds=preds, probs=probs)


if __name__ == "__main__":
    main()
