"""Pinecone-hosted label embeddings with a local cache, plus Pinecone storage and lookup."""

from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
import numpy as np  # type: ignore
from pinecone import Pinecone, ServerlessSpec  # type: ignore

MODEL = "llama-text-embed-v2"
DIMENSION = 1024
BATCH = 96  # documented max_batch_size; confirmed against the live model info at runtime
INDEX = "interest-matching"
METRIC = "cosine"
UPSERT_BATCH = 100
CACHE_PATH = Path(__file__).resolve().parent / "cache" / f"embeddings_{MODEL}_{DIMENSION}.npz"

_client: Pinecone | None = None
_cache: dict[str, np.ndarray] | None = None
_batch_size: int | None = None

def client() -> Pinecone:
    global _client
    if _client is None:
        _client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return _client

def batch_size() -> int:
    """Smaller of our default and the model's advertised limit; fails fast on an unsupported dimension."""
    global _batch_size
    if _batch_size is None:
        info = client().inference.get_model(model=MODEL)
        if info.supported_dimensions and DIMENSION not in info.supported_dimensions:
            raise ValueError(f"{MODEL} does not support dimension {DIMENSION}: {info.supported_dimensions}")
        _batch_size = min(BATCH, info.max_batch_size or BATCH)
    return _batch_size

def _key(text: str, input_type: str) -> str:
    return hashlib.sha1(f"{MODEL}|{DIMENSION}|{input_type}|{text}".encode("utf-8")).hexdigest()

def _load_cache() -> dict[str, np.ndarray]:
    global _cache
    if _cache is None:
        _cache = dict(np.load(CACHE_PATH)) if CACHE_PATH.is_file() else {}
    return _cache

def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_name(CACHE_PATH.stem + ".tmp.npz")
    np.savez(tmp, **_load_cache())
    tmp.replace(CACHE_PATH)

def _validated(batch: list[str], response) -> list[np.ndarray]:
    vectors = [np.asarray(item.values, dtype=np.float32) for item in response]
    if len(vectors) != len(batch):
        raise ValueError(f"embedding response has {len(vectors)} vectors for {len(batch)} inputs")
    out = []
    for text, vector in zip(batch, vectors):
        norm = float(np.linalg.norm(vector))
        if vector.shape != (DIMENSION,) or not np.isfinite(vector).all() or norm == 0.0:
            raise ValueError(f"invalid embedding for {text!r}: shape {vector.shape}, norm {norm}")
        out.append(vector / norm)
    return out

def embed(texts: list[str], input_type: str) -> np.ndarray:
    """L2-normalized float32 vectors, one row per text; only uncached texts hit the API."""
    cache = _load_cache()
    missing = list(dict.fromkeys(t for t in texts if _key(t, input_type) not in cache))
    try:
        if missing:
            size = batch_size()
            for start in range(0, len(missing), size):
                batch = missing[start : start + size]
                response = client().inference.embed(
                    model=MODEL,
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "END", "dimension": DIMENSION},
                )
                for text, vector in zip(batch, _validated(batch, response)):
                    cache[_key(text, input_type)] = vector
    finally:
        if missing:
            _save_cache()
    if not texts:
        return np.zeros((0, DIMENSION), dtype=np.float32)
    return np.stack([cache[_key(t, input_type)] for t in texts])

def index():
    """The matching index, created if absent; refuses an incompatible existing index and waits until ready."""
    pc = client()
    if not pc.has_index(INDEX):
        pc.create_index(  # blocks until the index reports ready
            name=INDEX,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    description = pc.describe_index(INDEX)
    if (description.dimension, description.metric) != (DIMENSION, METRIC):
        raise ValueError(
            f"index {INDEX!r} is {description.dimension}-dim/{description.metric}, expected {DIMENSION}-dim/{METRIC}"
        )
    for _ in range(60):
        if description.status.ready:
            return pc.Index(host=description.host)
        time.sleep(5)
        description = pc.describe_index(INDEX)
    raise TimeoutError(f"index {INDEX!r} not ready after 5 minutes")

def upsert(namespace: str, ids: list[str], vectors: np.ndarray, metadata: list[dict]) -> None:
    if not len(ids) == len(vectors) == len(metadata):
        raise ValueError(f"upsert length mismatch: {len(ids)} ids, {len(vectors)} vectors, {len(metadata)} metadata")
    if len(set(ids)) != len(ids):
        raise ValueError("upsert ids are not unique; later records would silently overwrite earlier ones")
    target = index()
    records = [{"id": i, "values": v.tolist(), "metadata": m} for i, v, m in zip(ids, vectors, metadata)]
    for start in range(0, len(records), UPSERT_BATCH):
        target.upsert(vectors=records[start : start + UPSERT_BATCH], namespace=namespace)

def query(namespace: str, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    result = index().query(namespace=namespace, vector=vector.tolist(), top_k=top_k)
    return [(match.id, match.score) for match in result.matches]