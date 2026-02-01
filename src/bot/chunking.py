from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ChunkRecord:
    chunk_id: int
    started_at_unix: float
    ended_at_unix: float
    duration_s: float
    chunk_dir: str
    audio_index: str


def chunks_path(session_dir: Path) -> Path:
    return session_dir / "chunks.json"


def load_chunks(session_dir: Path) -> List[Dict[str, Any]]:
    path = chunks_path(session_dir)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_chunks(session_dir: Path, items: List[Dict[str, Any]]) -> Path:
    path = chunks_path(session_dir)
    path.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
    return path


def next_chunk_id(session_dir: Path) -> int:
    items = load_chunks(session_dir)
    if not items:
        return 1
    last = items[-1].get("chunk_id", len(items))
    return int(last) + 1


def chunk_dir(session_dir: Path, chunk_id: int) -> Path:
    return session_dir / "chunks" / f"{chunk_id:04d}"


def append_chunk(session_dir: Path, chunk: ChunkRecord) -> Path:
    items = load_chunks(session_dir)
    items.append(asdict(chunk))
    return write_chunks(session_dir, items)
