# src/ood_sentiment_llm/embedding/openai_embedding.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from sklearn.model_selection import train_test_split

from ..config import EMBED_CFG, ARTIFACT_DIR, RANDOM_SEED
from ..utils import ensure_dir, get_data_dir
from ..data.jsonl_io import load_jsonl_files


# =========================
# 1. 토큰 자르기
# =========================

_enc = tiktoken.get_encoding("cl100k_base")


def truncate_by_tokens(text: str, max_tokens: int | None = None) -> str:
    """
    tiktoken 기준으로 max_tokens 까지만 남기고 잘라주는 함수.
    """
    if max_tokens is None:
        max_tokens = EMBED_CFG.max_tokens

    toks = _enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _enc.decode(toks[:max_tokens])


# =========================
# 2. OpenAI Embedding 클라이언트
# =========================

def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    client = OpenAI(api_key=api_key)
    return client


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str | None = None,
) -> np.ndarray:
    """
    OpenAI Embeddings API 호출.

    Parameters
    ----------
    client : OpenAI
    texts  : str list
    model  : 사용할 embedding 모델명 (None이면 EMBED_CFG.model_name)

    Returns
    -------
    np.ndarray, shape (N, dim)
    """
    if model is None:
        model = EMBED_CFG.model_name

    safe_texts = [t if (t and t.strip()) else " " for t in texts]
    resp = client.embeddings.create(model=model, input=safe_texts)
    embs = [d.embedding for d in resp.data]
    return np.asarray(embs, dtype=np.float32)


# =========================
# 3. JSONL 로드 & 전처리
# =========================

def find_default_data_files() -> List[Path]:
    """
    기본 데이터 파일 목록 찾기.

    1) EMBED_DATA_FILES 환경변수가 있으면, 세미콜론/콤마로 구분된 경로 사용
    2) 없으면 data/raw 아래의 *.jsonl 전부 사용
    """
    env_val = os.environ.get("EMBED_DATA_FILES")
    if env_val:
        raw_paths = [x.strip() for x in env_val.replace(";", ",").split(",") if x.strip()]
        return [Path(p).expanduser().resolve() for p in raw_paths]

    raw_dir = get_data_dir("raw")
    paths = sorted(raw_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다. {raw_dir} 아래에 jsonl 파일을 두거나,\n"
            "환경변수 EMBED_DATA_FILES 로 경로를 지정하세요."
        )
    return paths


def build_clean_records(
    max_tokens: int | None = None,
) -> Tuple[list[dict], list[dict]]:
    """
    JSONL 파일들을 로드해서
    - text 토큰 길이 제한
    - group == 'N' / 'P' / 'OOD' 기준으로 PN / OOD 분리.

    Returns
    -------
    pn_records : group ∈ {N, P}
    ood_records: group == OOD
    """
    data_files = find_default_data_files()
    print("[openai_embedding] 데이터 파일:")
    for p in data_files:
        print("  -", p)

    raw = load_jsonl_files(data_files)
    print(f"[openai_embedding] 총 레코드 수: {len(raw)}")

    cleaned: list[dict] = []
    for r in raw:
        txt = truncate_by_tokens(r["text"], max_tokens=max_tokens)
        g_raw = r.get("group")
        if g_raw is None:
            # group 정보 없으면 OOD/PN 판단 불가 → 스킵
            continue
        g = str(g_raw).upper()
        if g not in {"N", "P", "OOD"}:
            continue
        cleaned.append({"text": txt, "group": g})

    pn_records = [x for x in cleaned if x["group"] in {"N", "P"}]
    ood_records = [x for x in cleaned if x["group"] == "OOD"]

    print(f"[openai_embedding] PN(=N/P): {len(pn_records)} | OOD: {len(ood_records)}")
    return pn_records, ood_records


def build_train_test_split(
    pn_records: list[dict],
    test_size: float = 0.2,
) -> Tuple[list[dict], list[dict]]:
    """
    PN 레코드를 train/test로 나눈다. (stratify=group)
    """
    n = len(pn_records)
    if n == 0:
        raise ValueError("PN 레코드가 0개입니다. group ∈ {N, P} 데이터가 필요합니다.")

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=[r["group"] for r in pn_records],
    )
    train_records = [pn_records[i] for i in train_idx]
    test_records = [pn_records[i] for i in test_idx]

    print(
        f"[openai_embedding] train/test split: "
        f"{len(train_records)} train / {len(test_records)} test"
    )
    return train_records, test_records


# =========================
# 4. 임베딩 인코딩
# =========================

def encode_records(
    client: OpenAI,
    records: list[dict],
    batch_size: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PN 레코드들을 임베딩 + 라벨(0/1)로 변환.

    group: 'N' -> 0, 'P' -> 1
    """
    if batch_size is None:
        batch_size = EMBED_CFG.batch_size

    texts = [r["text"] for r in records]
    y = np.asarray(
        [0 if r["group"] == "N" else 1 for r in records],
        dtype=np.int64,
    )

    emb_list: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding PN"):
        chunk = texts[i : i + batch_size]
        embs = embed_texts(client, chunk)
        emb_list.append(embs)

    X = np.vstack(emb_list) if emb_list else np.zeros((0, EMBED_CFG.dim), dtype=np.float32)
    return X, y


def encode_ood_records(
    client: OpenAI,
    records: list[dict],
    batch_size: int | None = None,
) -> np.ndarray:
    """
    OOD 레코드들을 임베딩으로 변환.
    """
    if batch_size is None:
        batch_size = EMBED_CFG.batch_size

    texts = [r["text"] for r in records]
    emb_list: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding OOD"):
        chunk = texts[i : i + batch_size]
        embs = embed_texts(client, chunk)
        emb_list.append(embs)

    X = np.vstack(emb_list) if emb_list else np.zeros((0, EMBED_CFG.dim), dtype=np.float32)
    return X


# =========================
# 5. 옛 인터페이스용 헬퍼
# =========================

def build_pn_ood_embeddings_from_jsonl(
    test_size: float = 0.2,
    max_tokens: int | None = None,
    return_records: bool = False,
):
    """
    (기존 코드 호환용) JSONL에서 바로 PN/OOD 임베딩을 만들어 반환.

    Returns
    -------
    X_train, y_train, X_test, y_test, X_ood
    (+ return_records=True 이면 레코드들도 함께 반환)
    """
    pn_records, ood_records = build_clean_records(
        max_tokens=max_tokens if max_tokens is not None else EMBED_CFG.max_tokens
    )
    train_records, test_records = build_train_test_split(
        pn_records,
        test_size=test_size,
    )

    client = _build_client()

    X_train, y_train = encode_records(client, train_records)
    X_test, y_test = encode_records(client, test_records)
    X_ood = encode_ood_records(client, ood_records)

    if return_records:
        return (
            X_train,
            y_train,
            X_test,
            y_test,
            X_ood,
            train_records,
            test_records,
            ood_records,
        )
    else:
        return X_train, y_train, X_test, y_test, X_ood


# =========================
# 6. main() : artifacts에 .npy 저장
# =========================

def main(
    test_size: float = 0.2,
) -> None:
    """
    전체 임베딩 파이프라인:

    1) JSONL 로드 & 토큰 길이 제한
    2) PN(=N/P) vs OOD 레코드 분리
    3) PN을 train/test로 split
    4) OpenAI 임베딩 추출
    5) artifacts 디렉터리에 저장

       - embeddings_train_X.npy
       - embeddings_train_y.npy
       - embeddings_test_X.npy
       - embeddings_test_y.npy
       - embeddings_ood_test_X.npy
       - embeddings_IND_X.npy
       - embeddings_OOD_X.npy
    """
    np.random.seed(RANDOM_SEED)
    ensure_dir(ARTIFACT_DIR)

    pn_records, ood_records = build_clean_records(max_tokens=EMBED_CFG.max_tokens)
    train_records, test_records = build_train_test_split(pn_records, test_size=test_size)

    client = _build_client()

    print("[openai_embedding] PN(train) 임베딩 추출...")
    X_train, y_train = encode_records(client, train_records)

    print("[openai_embedding] PN(test) 임베딩 추출...")
    X_test, y_test = encode_records(client, test_records)

    print("[openai_embedding] OOD 임베딩 추출...")
    X_ood = encode_ood_records(client, ood_records)

    np.save(ARTIFACT_DIR / "embeddings_train_X.npy", X_train)
    np.save(ARTIFACT_DIR / "embeddings_train_y.npy", y_train)

    np.save(ARTIFACT_DIR / "embeddings_test_X.npy", X_test)
    np.save(ARTIFACT_DIR / "embeddings_test_y.npy", y_test)

    np.save(ARTIFACT_DIR / "embeddings_ood_test_X.npy", X_ood)

    # OOD 실험용 별도 이름
    np.save(ARTIFACT_DIR / "embeddings_IND_X.npy", X_test)
    np.save(ARTIFACT_DIR / "embeddings_OOD_X.npy", X_ood)

    print("[openai_embedding] 저장 완료:")
    print("  - embeddings_train_X.npy", X_train.shape)
    print("  - embeddings_train_y.npy", y_train.shape)
    print("  - embeddings_test_X.npy ", X_test.shape)
    print("  - embeddings_test_y.npy ", y_test.shape)
    print("  - embeddings_ood_test_X.npy", X_ood.shape)


if __name__ == "__main__":
    main()
