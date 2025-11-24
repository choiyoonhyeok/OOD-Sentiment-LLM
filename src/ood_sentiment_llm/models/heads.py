from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    간단한 2층 MLP 투영기.
    입력 차원(in_dim)을 proj_dim 차원으로 투영한 뒤 다시 proj_dim으로 변환.
    """
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    """
    ProjectionHead 출력(proj_dim)을 받아 2클래스 로짓을 뽑는 선형 레이어.
    """
    def __init__(self, proj_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(proj_dim, num_classes)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        return self.fc(fx)
