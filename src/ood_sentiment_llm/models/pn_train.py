from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..config import PN_CFG, ARTIFACT_DIR, RANDOM_SEED
from ..utils import ensure_dir
from .heads import ProjectionHead, ClassifierHead


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingDataset(Dataset):
    """
    고정 LLM 임베딩을 받아 학습에 사용하는 Dataset.
    y: 0(neg), 1(pos)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def prototype_loss(
    fx: torch.Tensor,
    y: torch.Tensor,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
) -> torch.Tensor:
    """
    L_pl = || f(x) - m_c ||^2 의 배치 평균
    - fx:        (B, D) 투영 임베딩
    - y:         (B,)   레이블 0/1
    - proto_neg: (D,)   neg 프로토타입
    - proto_pos: (D,)   pos 프로토타입
    """
    target_proto = torch.where(
        (y == 0).unsqueeze(-1),       # (B,1)
        proto_neg.unsqueeze(0),       # (1,D) -> (B,D)
        proto_pos.unsqueeze(0),       # (1,D) -> (B,D)
    )
    loss = (fx - target_proto).pow(2).sum(dim=1).mean()
    return loss


def update_prototypes_ema(
    fx: torch.Tensor,
    y: torch.Tensor,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
    ema: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    현재 배치의 클래스별 평균(fx)을 사용해 전역 프로토타입을 EMA로 갱신.
    """
    if (y == 0).any():
        batch_mean_neg = fx[y == 0].mean(dim=0)
        proto_neg[:] = ema * proto_neg + (1.0 - ema) * batch_mean_neg
    if (y == 1).any():
        batch_mean_pos = fx[y == 1].mean(dim=0)
        proto_pos[:] = ema * proto_pos + (1.0 - ema) * batch_mean_pos
    return proto_neg, proto_pos


def evaluate(model_proj: nn.Module, model_clf: nn.Module, loader: DataLoader) -> float:
    model_proj.eval()
    model_clf.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            fx = model_proj(X)
            logits = model_clf(fx)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
    return correct / max(1, total)


def train_loop(
    model_proj: nn.Module,
    model_clf: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    proto_neg: torch.Tensor,
    proto_pos: torch.Tensor,
    artifact_dir: Path,
) -> None:
    ce_criterion = nn.CrossEntropyLoss()
    params = list(model_proj.parameters()) + list(model_clf.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=PN_CFG.lr,
        weight_decay=PN_CFG.weight_decay,
    )

    best_val_acc = -1.0

    for epoch in range(1, PN_CFG.epochs + 1):
        model_proj.train()
        model_clf.train()

        running_ce = 0.0
        running_pl = 0.0
        running_acc = 0.0
        n_samples = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            fx = model_proj(X)               # (B, D)
            logits = model_clf(fx)           # (B, 2)

            loss_ce = ce_criterion(logits, y)
            loss_pl = prototype_loss(fx, y, proto_neg, proto_pos)
            loss = loss_ce + PN_CFG.lambda_pl * loss_pl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 프로토타입 EMA 업데이트
            proto_neg, proto_pos = update_prototypes_ema(
                fx.detach(), y.detach(), proto_neg, proto_pos, ema=PN_CFG.proto_ema
            )

            bs = X.size(0)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

            running_ce += loss_ce.item() * bs
            running_pl += loss_pl.item() * bs
            running_acc += acc * bs
            n_samples += bs

        epoch_ce = running_ce / max(1, n_samples)
        epoch_pl = running_pl / max(1, n_samples)
        epoch_acc = running_acc / max(1, n_samples)

        val_acc = evaluate(model_proj, model_clf, val_loader)

        print(
            f"[Epoch {epoch}] "
            f"Train Acc={epoch_acc:.4f} | CE={epoch_ce:.4f} | PL={epoch_pl:.4f} || "
            f"Val Acc={val_acc:.4f}"
        )

        # 최고 성능 갱신 시 체크포인트 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ensure_dir(artifact_dir)
            ckpt = {
                "proj": model_proj.state_dict(),
                "clf": model_clf.state_dict(),
                "proto_neg": proto_neg.detach().cpu().numpy(),
                "proto_pos": proto_pos.detach().cpu().numpy(),
                "proj_dim": model_proj.net[0].out_features,
            }
            torch.save(ckpt, artifact_dir / "best_pn_classifier.pt")
            print(f"  ↳ 모델 저장(Val 최고): {best_val_acc:.4f}")

    print(f"학습 완료. 최종 최고 Val Acc={best_val_acc:.4f}")


def main() -> None:
    # 시드 고정
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    ART = ARTIFACT_DIR

    train_x_path = ART / "embeddings_train_X.npy"
    train_y_path = ART / "embeddings_train_y.npy"

    if not train_x_path.exists() or not train_y_path.exists():
        msg = (
            "학습용 임베딩 파일을 찾을 수 없습니다.\n"
            f"- {train_x_path}\n"
            f"- {train_y_path}\n"
            "OpenAI 임베딩 스크립트로 embeddings_train_X.npy / embeddings_train_y.npy 를 먼저 생성해 주세요."
        )
        raise FileNotFoundError(msg)

    X = np.load(train_x_path)  # (N, D)
    y = np.load(train_y_path)  # (N,)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )

    train_ds = EmbeddingDataset(X_train, y_train)
    val_ds = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=PN_CFG.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=PN_CFG.batch_size, shuffle=False
    )

    in_dim = X.shape[1]
    proj_dim = PN_CFG.proj_dim

    model_proj = ProjectionHead(in_dim=in_dim, proj_dim=proj_dim).to(DEVICE)
    model_clf = ClassifierHead(proj_dim=proj_dim, num_classes=2).to(DEVICE)

    proto_neg = torch.zeros(proj_dim, dtype=torch.float32, device=DEVICE, requires_grad=False)
    proto_pos = torch.zeros(proj_dim, dtype=torch.float32
