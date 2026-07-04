"""Vector search over successful fixes. Retrieves few-shot examples for the specialist prompt."""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

log = logging.getLogger("mlfix.semantic")


# We import heavy deps lazily so daemon startup stays fast when the DB is empty
_lancedb = None
_embedder = None
_embedder_lock = Lock()


def _get_lancedb():
    global _lancedb
    if _lancedb is None:
        import lancedb
        _lancedb = lancedb
    return _lancedb


def _get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            log.info("loading embedding model (first run downloads ~80MB)...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("embedding model loaded")
    return _embedder


class SemanticMemory:
    TABLE = "fixes"
    DIM = 384  # all-MiniLM-L6-v2 output dim

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._table = None

    def _connect(self):
        if self._db is None:
            lancedb = _get_lancedb()
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _get_table(self):
        if self._table is not None:
            return self._table
        db = self._connect()
        try:
            self._table = db.open_table(self.TABLE)
        except Exception:
            # Table doesn't exist yet; create empty
            import pyarrow as pa
            schema = pa.schema([
                pa.field("episode_id", pa.string()),
                pa.field("category", pa.string()),
                pa.field("error", pa.string()),
                pa.field("original_code", pa.string()),
                pa.field("fixed_code", pa.string()),
                pa.field("explanation", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.DIM)),
            ])
            self._table = db.create_table(self.TABLE, schema=schema)
        return self._table

    def _embed(self, text: str) -> list[float]:
        emb = _get_embedder()
        return emb.encode(text, normalize_embeddings=True).tolist()

    def _key(self, error: str, code: str) -> str:
        # What we embed. Error dominates because that's what we retrieve on.
        return f"ERROR: {error.strip()[:800]}\nCODE HEAD: {code[:400]}"

    def add(
        self,
        episode_id: str,
        category: str,
        error: str,
        original_code: str,
        fixed_code: str,
        explanation: str,
    ) -> None:
        try:
            vec = self._embed(self._key(error, original_code))
            self._get_table().add([{
                "episode_id": episode_id,
                "category": category,
                "error": error[:2000],
                "original_code": original_code[:4000],
                "fixed_code": fixed_code[:4000],
                "explanation": explanation[:1000],
                "vector": vec,
            }])
            log.info("added semantic entry %s (category=%s)", episode_id, category)
        except Exception as e:
            log.warning("failed to add semantic entry: %s", e)

    def retrieve(
        self,
        error: str,
        code: str,
        category: str | None = None,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return up to k most similar past fixes. Filter by category if given."""
        try:
            tbl = self._get_table()
            if len(tbl) == 0:
                return []
            vec = self._embed(self._key(error, code))
            q = tbl.search(vec).limit(k * 3)  # over-fetch, filter, then trim
            results = q.to_list()
            if category:
                results = [r for r in results if r.get("category") == category]
            return results[:k]
        except Exception as e:
            log.warning("semantic retrieve failed: %s", e)
            return []