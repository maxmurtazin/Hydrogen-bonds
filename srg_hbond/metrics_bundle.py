from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, IO, Iterator, List, Optional


def append_metrics_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    """Append one JSON object per line (JSONL). Uses default=str for non-JSON-native values."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def load_metrics_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iter_metrics_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class MetricsJsonlWriter:
    """Append-only JSONL stream for step metrics (shared by demos and offline figure generation)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[IO[str]] = self.path.open("w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("MetricsJsonlWriter is closed")
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> MetricsJsonlWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
