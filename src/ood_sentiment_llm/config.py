from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import get_project_root, get_data_dir, get_artifact_dir


# =========================
# 경로 설정
# =========================

PROJECT_ROOT: Path = get_project_root()
DATA_DIR: Path = get_data_dir()             # <root>/data
DATA_RAW_DIR: Path = get_data_dir("raw")    # <root>/data/raw
DATA_PROCESSED_DIR: Path = get_data_dir("processed")
ARTIFACT_DIR: Path = get_artifact_dir()     # <root>/artifacts

RANDOM_SEED: int = 42


# =========================
# 임베딩 설정
# =========================

@dataclass
class EmbeddingConfig:
    """
    OpenAI 임베딩 관련 기본 설정.
    """
    model_name: str = "text-embedding-3-large"
    dim: int = 1536
    batch_size: int = 256
    max_tokens: int = 512          # tiktoken 으로 자를 최대 토큰 길이
    truncate_long_text: bool = True


# =========================
# PN 분류기 설정
# =========================

@dataclass
class PNClassifierConfig:
    """
    Projection + Prototype + Classifier 구조 학습 설정.
    """
    proj_dim: int = 256
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 10
    lambda_pl: float = 0.5      # Prototype Loss 비중
    proto_ema: float = 0.9      # 프로토타입 EMA 계수
    weight_decay: float = 0.0   # 필요 시 L2 정규화 추가용


EMBED_CFG = EmbeddingConfig()
PN_CFG = PNClassifierConfig()
