from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    현재 파일(src/ood_sentiment_llm/utils/paths.py) 위치를 기준으로
    프로젝트 루트(OOD-Sentiment-LLM)를 찾아 반환.
    """
    return Path(__file__).resolve().parents[2]


def get_src_dir() -> Path:
    """src 디렉터리 경로 (…/OOD-Sentiment-LLM/src)."""
    return get_project_root() / "src"


def get_data_dir(subdir: str | None = None) -> Path:
    """
    data 디렉터리 경로.
    - 기본: <root>/data
    - subdir 지정: <root>/data/<subdir>
    """
    base = get_project_root() / "data"
    return base / subdir if subdir is not None else base


def get_artifact_dir() -> Path:
    """학습 결과/임베딩/모델 등을 저장하는 artifacts 디렉터리."""
    return get_project_root() / "artifacts"


def ensure_dir(path: Path) -> Path:
    """디렉터리가 없으면 생성한 뒤 그대로 반환."""
    path.mkdir(parents=True, exist_ok=True)
    return path
