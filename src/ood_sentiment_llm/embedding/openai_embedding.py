from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tiktoken
from tqdm import tqdm
from openai import OpenAI

from ..config import EMBED_CFG, ARTIFACT_DIR
from ..utils import ensure_dir
from ..data import load_jsonl_files


# --------------------------------------------------------------------
# 토크나이저 & 텍스트 자르기
# --------------------------------------------------------------------

_tokenizer_cache: dict[str, tiktoken.Encoding] = {}


def _get_tokenizer(model_name: str) -> tiktoken.Encoding:
    """
    OpenAI 임베딩/챗 모델 이름을 받아 적당한 tiktoken encoding 리턴.
    못 찾으면 cl100k_base 로 fallback.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    _tokenizer_cache[model_name] = enc
    return enc


def truncate_by_tokens(
    text: str,
    max_tokens: int | None = None,
    model_name: str | None = None,
) -> str:
    """
    토큰 길이가 max_tokens 를 넘으면 잘라서 반환.
    - 주로 JSONL → 임베딩 전에 사용.
    """
    if max_tokens is None or max_tokens <= 0:
        return text

    model_name = model_name or EMBED_CFG.model_name
    enc = _get_tokenizer(model_name)

    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text

    tokens = tokens[:max_tokens]
    return enc.decode(tokens)


# --------------------------------------------------------------------
# OpenAI Embeddings 호출
# --------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # OPENAI_API_KEY 는 환경변수로 설정되어 있어야 함
        _client = OpenAI()
    return _client


def embed_texts(
    texts: List[str],
    model_name: str | None = None,
) -> np.ndarray:
    """
    여러 문장을 한 번에 임베딩.
    - model_name 이 None 이면 config.EMBED_CFG.model_name 사용.
    - 반환 shape: (N, dim)
    """
    if not texts:
        return np.zeros((0, EMBED_CFG.dim), dtype=np.float32)

    model_name = model_name or EMBED_CFG.model_name
    client = _get_client()

    resp = client.embeddings.create(
        model=model_name,
        input=texts,
    )
    # resp.data[i].embedding : List[float]
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return arr


# --------------------------------------------------------------------
# 간단한 end-to-end CLI 예시 (선택)
# --------------------------------------------------------------------

def build_pn_ood_embeddings_from_jsonl(
    file_paths: Iterable[str | Path],
    max_tokens: int | None = None,
    batch_size: int | None = None,
    artifact_dir: Path | None = None,
) -> None:
    """
    JSONL 파일들에서
      - group == 'N' / 'P' 인 것 → PN(IND) 용 train 임베딩
      - group == 'OOD' 인 것 → OOD 임베딩
    을 만들어 artifacts 폴더에 저장.

    저장 파일:
      - embeddings_train_X.npy, embeddings_train_y.npy
      - embeddings_ood_train_X.npy
    """
    artifact_dir = artifact_dir or ARTIFACT_DIR
    ensure_dir(artifact_dir)

    max_tokens = max_tokens or EMBED_CFG.max_tokens
    batch_size = batch_size or EMBED_CFG.batch_size

    records = load_jsonl_files(file_paths)
    print(f"[embedding] 총 로우 수: {len(records)}")

    pn_texts: list[str] = []
    pn_labels: list[int] = []  # 0=neg(N), 1=pos(P)
    ood_texts: list[str] = []

    for r in records:
        g = str(r.get("group", "")).upper()
        txt = truncate_by_tokens(r["text"], max_tokens)
        if g == "N":
            pn_texts.append(txt)
            pn_labels.append(0)
        elif g == "P":
            pn_texts.append(txt)
            pn_labels.append(1)
        elif g == "OOD":
            ood_texts.append(txt)
        else:
            # 모르는 group 은 그냥 버림
            continue

    print(f"  PN(IND) 텍스트: {len(pn_texts)} | OOD 텍스트: {len(ood_texts)}")

    # --- PN 임베딩 ---
    print("[embedding] PN(IND) 임베딩 추출...")
    pn_emb_list: list[np.ndarray] = []
    for i in tqdm(range(0, len(pn_texts), batch_size)):
        chunk = pn_texts[i : i + batch_size]
        embs = embed_texts(chunk)
        pn_emb_list.append(embs)
    pn_X = np.vstack(pn_emb_list) if pn_emb_list else np.zeros(
        (0, EMBED_CFG.dim), dtype=np.float32
    )
    pn_y = np.array(pn_labels, dtype=np.int64)

    # --- OOD 임베딩 ---
    print("[embedding] OOD 임베딩 추출...")
    ood_emb_list: list[np.ndarray] = []
    for i in tqdm(range(0, len(ood_texts), batch_size)):
        chunk = ood_texts[i : i + batch_size]
        embs = embed_texts(chunk)
        ood_emb_list.append(embs)
    ood_X = np.vstack(ood_emb_list) if ood_emb_list else np.zeros(
        (0, EMBED_CFG.dim), dtype=np.float32
    )

    # --- 저장 ---
    print("[embedding] 저장...")
    np.save(artifact_dir / "embeddings_train_X.npy", pn_X)
    np.save(artifact_dir / "embeddings_train_y.npy", pn_y)
    np.save(artifact_dir / "embeddings_ood_train_X.npy", ood_X)
    print(f"  → {artifact_dir} 에 임베딩 저장 완료")


def main() -> None:
    """
    간단 테스트용 엔트리포인트.
    - 환경변수 DATA_FILES 에 콤마(,)로 구분된 jsonl 경로들을 넣어두고 실행.
      예) DATA_FILES="data/raw/A.jsonl,data/raw/B.jsonl"
    """
    files_env = os.environ.get("DATA_FILES", "")
    if not files_env:
        raise SystemExit(
            "환경변수 DATA_FILES 가 비어 있습니다.\n"
            "예시: DATA_FILES='data/raw/A.jsonl,data/raw/B.jsonl'"
        )

    file_paths = [s.strip() for s in files_env.split(",") if s.strip()]
    build_pn_ood_embeddings_from_jsonl(file_paths)


if __name__ == "__main__":
    main()
