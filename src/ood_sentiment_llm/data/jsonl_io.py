from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

try:
    import ujson as json  # 빠르면 ujson 우선
except ImportError:       # pragma: no cover
    import json           # type: ignore


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    JSONL 파일 한 줄씩 읽어서 dict 로 yield.
    - 비어 있거나 파싱 실패한 라인은 조용히 스킵.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def load_jsonl_files(file_paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    """
    여러 JSONL 파일을 한 번에 로드해서 리스트로 반환.

    - 최소한 "text" 필드가 있는 object 만 사용.
    - "group" 필드가 있으면 그대로 보존 (N / P / OOD 등).
    """
    records: List[Dict[str, Any]] = []
    for p in file_paths:
        for obj in iter_jsonl(Path(p)):
            if "text" not in obj:
                continue
            rec: Dict[str, Any] = {"text": obj["text"]}
            if "group" in obj:
                rec["group"] = obj["group"]
            records.append(rec)
    return records
