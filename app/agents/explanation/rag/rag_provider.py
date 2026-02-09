import json
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from app.agents.explanation.rag.corpus_registry import AVAILABLE_CORPORA, CorpusEntry
from app.agents.explanation.rag.retrieval_contract import Reference, RetrievalRequest
from app.core.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MAX_REFERENCES = 3
CHUNKER_VERSION = "sentence_v1"
DEFAULT_CONTEXT_WINDOW = 4

RAG_RETRIEVER_MODE_ENV = "RAG_RETRIEVER_MODE"
DEFAULT_RETRIEVER_MODE = "hybrid"

EMBEDDING_MODEL_ENV = "OPENAI_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_TOP_K_ENV = "EMBEDDING_TOP_K"
EMBEDDING_THRESHOLD_ENV = "EMBEDDING_THRESHOLD"
RAG_CACHE_PATH_ENV = "RAG_CACHE_PATH"
DEFAULT_EMBEDDING_TOP_K = 10
DEFAULT_EMBEDDING_THRESHOLD = 0.25
EMBEDDING_BATCH_SIZE = 64

RAG_CONTEXT_WINDOW_ENV = "RAG_CONTEXT_WINDOW"

_OAI_CLIENT: Optional[OpenAI] = None
_CORPUS_CHUNKS: Optional[List["CorpusChunk"]] = None
_CORPUS_FINGERPRINT: Optional[str] = None
_EMBEDDING_INDEX: Optional[Dict[str, List[float]]] = None
_EMBEDDING_INDEX_MODEL: Optional[str] = None
_EMBEDDING_INDEX_FINGERPRINT: Optional[str] = None


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    source_title: str
    text: str
    tags: List[str]
    path: str


def _get_openai_client() -> OpenAI:
    global _OAI_CLIENT
    if _OAI_CLIENT is None:
        _OAI_CLIENT = OpenAI()
    return _OAI_CLIENT


def _discover_corpus_entries() -> List[CorpusEntry]:
    entries = list(AVAILABLE_CORPORA)
    seen = {Path(entry.path).name for entry in entries}
    try:
        for file_path in CORPUS_DIR.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() != ".jsonl":
                continue
            if file_path.name in seen:
                continue
            entries.append(
                CorpusEntry(
                    source=file_path.stem,
                    note=file_path.stem,
                    path=file_path.name,
                    tags=[],
                )
            )
    except FileNotFoundError:
        return entries
    return entries


def _resolve_path(entry: CorpusEntry) -> Path:
    path = Path(entry.path)
    if path.is_absolute():
        return path
    candidate = CORPUS_DIR / path
    if candidate.exists():
        return candidate

    target = unicodedata.normalize("NFC", candidate.name)
    for file_path in CORPUS_DIR.iterdir():
        if not file_path.is_file():
            continue
        if unicodedata.normalize("NFC", file_path.name) == target:
            return file_path
        if unicodedata.normalize("NFD", file_path.name) == unicodedata.normalize(
            "NFD", target
        ):
            return file_path
    return candidate


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


def _tfidf_vectors(texts: List[str]) -> List[Dict[str, float]]:
    tokenized = [_tokenize(text) for text in texts]
    doc_counts = [Counter(tokens) for tokens in tokenized]
    df = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    total_docs = len(texts)
    idf = {
        term: math.log((1 + total_docs) / (1 + freq)) + 1.0 for term, freq in df.items()
    }

    vectors: List[Dict[str, float]] = []
    for counts in doc_counts:
        total = sum(counts.values())
        if total == 0:
            vectors.append({})
            continue
        vec = {term: (count / total) * idf.get(term, 0.0) for term, count in counts.items()}
        vectors.append(vec)
    return vectors


def _cosine_similarity_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cosine_similarity_dense(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tag_overlap(query_tokens: List[str], tags: List[str]) -> int:
    if not query_tokens or not tags:
        return 0
    normalized_tags = [tag.lower() for tag in tags]
    count = 0
    for token in query_tokens:
        token_lower = token.lower()
        for tag in normalized_tags:
            if token_lower in tag or tag in token_lower:
                count += 1
                break
    return count


def _normalize_tags(raw_tags: List[object]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for tag in raw_tags:
        text = str(tag).strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _source_title(entry: CorpusEntry) -> str:
    path = _resolve_path(entry)
    if path.suffix:
        return path.stem
    return path.name if path.name else entry.path


def _compute_corpus_fingerprint(entries: List[CorpusEntry]) -> str:
    hasher = sha256()
    hasher.update(CHUNKER_VERSION.encode("utf-8"))
    for entry in entries:
        hasher.update(str(entry.path).encode("utf-8"))
        hasher.update("|".join(entry.tags).encode("utf-8"))
        path = _resolve_path(entry)
        try:
            hasher.update(path.read_bytes())
        except FileNotFoundError:
            continue
    return hasher.hexdigest()


def _load_jsonl_chunks(entry: CorpusEntry) -> List[CorpusChunk]:
    path = _resolve_path(entry)
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        return []

    source_title = _source_title(entry)
    chunks: List[CorpusChunk] = []
    with handle:
        for line_idx, line in enumerate(handle):
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            metadata = item.get("metadata")
            tags: List[object] = list(entry.tags)
            if isinstance(metadata, dict):
                tags.extend(metadata.get("tags") or [])
                tags.extend(metadata.get("risk_signals") or [])
                domain = metadata.get("domain")
                if domain:
                    tags.append(domain)
            normalized_tags = _normalize_tags(tags)
            chunk_id = f"{source_title}::jsonl::{line_idx}"
            chunks.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    source_title=source_title,
                    text=text,
                    tags=normalized_tags,
                    path=str(entry.path),
                )
            )
    return chunks


def _load_corpus_chunks(entries: List[CorpusEntry]) -> List[CorpusChunk]:
    chunks: List[CorpusChunk] = []
    for entry in entries:
        path = _resolve_path(entry)
        if path.suffix.lower() != ".jsonl":
            continue
        chunks.extend(_load_jsonl_chunks(entry))
    return chunks


def _parse_chunk_index(chunk: CorpusChunk) -> Optional[int]:
    parts = chunk.chunk_id.split("::")
    if len(parts) >= 3 and parts[-2] in {"sentence", "jsonl"}:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


def _index_chunks_by_source(
    chunks: List[CorpusChunk],
) -> Tuple[Dict[str, List[Tuple[int, CorpusChunk]]], Dict[str, Dict[int, int]]]:
    by_source: Dict[str, List[Tuple[int, CorpusChunk]]] = {}
    index_map: Dict[str, Dict[int, int]] = {}
    for chunk in chunks:
        idx = _parse_chunk_index(chunk)
        if idx is None:
            continue
        by_source.setdefault(chunk.source_title, []).append((idx, chunk))

    for source_title, pairs in by_source.items():
        pairs.sort(key=lambda item: item[0])
        index_map[source_title] = {idx: pos for pos, (idx, _) in enumerate(pairs)}

    return by_source, index_map


def _expand_chunk_context(
    chunk: CorpusChunk,
    by_source: Dict[str, List[Tuple[int, CorpusChunk]]],
    index_map: Dict[str, Dict[int, int]],
    window: int,
) -> str:
    if window <= 0:
        return chunk.text
    idx = _parse_chunk_index(chunk)
    if idx is None:
        return chunk.text
    pairs = by_source.get(chunk.source_title)
    if not pairs:
        return chunk.text
    pos = index_map.get(chunk.source_title, {}).get(idx)
    if pos is None:
        return chunk.text
    start = max(0, pos - window)
    end = min(len(pairs), pos + window + 1)
    return " ".join(pair[1].text for pair in pairs[start:end])


def _get_corpus_chunks_and_fingerprint() -> Tuple[List[CorpusChunk], str]:
    global _CORPUS_CHUNKS, _CORPUS_FINGERPRINT
    if _CORPUS_CHUNKS is not None and _CORPUS_FINGERPRINT is not None:
        return _CORPUS_CHUNKS, _CORPUS_FINGERPRINT

    entries = _discover_corpus_entries()
    fingerprint = _compute_corpus_fingerprint(entries)
    chunks = _load_corpus_chunks(entries)

    _CORPUS_CHUNKS = chunks
    _CORPUS_FINGERPRINT = fingerprint
    return chunks, fingerprint


def _get_retriever_mode() -> str:
    raw = os.getenv(RAG_RETRIEVER_MODE_ENV, DEFAULT_RETRIEVER_MODE).strip().lower()
    if raw in {"tfidf", "embedding", "hybrid"}:
        return raw
    return DEFAULT_RETRIEVER_MODE


def _get_embedding_model() -> str:
    return os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def _get_embedding_top_k() -> int:
    raw = os.getenv(EMBEDDING_TOP_K_ENV, str(DEFAULT_EMBEDDING_TOP_K))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_EMBEDDING_TOP_K
    return value if value > 0 else DEFAULT_EMBEDDING_TOP_K


def _get_embedding_threshold() -> float:
    raw = os.getenv(EMBEDDING_THRESHOLD_ENV, str(DEFAULT_EMBEDDING_THRESHOLD))
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_EMBEDDING_THRESHOLD
    return value if value > 0 else DEFAULT_EMBEDDING_THRESHOLD


def _get_context_window() -> int:
    raw = os.getenv(RAG_CONTEXT_WINDOW_ENV, str(DEFAULT_CONTEXT_WINDOW))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CONTEXT_WINDOW
    return value if value > 0 else 0


def _get_cache_path() -> Optional[Path]:
    raw = os.getenv(RAG_CACHE_PATH_ENV)
    if raw is not None and not raw.strip():
        return None
    if raw:
        return Path(raw)
    return CORPUS_DIR / ".embedding_cache.json"


def _load_embedding_cache(
    path: Path, fingerprint: str, model: str
) -> Optional[Dict[str, List[float]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    if payload.get("fingerprint") != fingerprint:
        return None
    if payload.get("model") != model:
        return None

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, dict):
        return None
    return embeddings


def _save_embedding_cache(
    path: Path, fingerprint: str, model: str, embeddings: Dict[str, List[float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "model": model,
        "chunker_version": CHUNKER_VERSION,
        "embeddings": embeddings,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = _get_openai_client()
    model = _get_embedding_model()
    response = client.embeddings.create(model=model, input=texts)
    data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in data]


def _compute_embeddings(chunks: List[CorpusChunk]) -> Dict[str, List[float]]:
    embeddings: Dict[str, List[float]] = {}
    texts = [chunk.text for chunk in chunks]
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = _embed_texts(batch_texts)
        for chunk, vector in zip(chunks[start : start + EMBEDDING_BATCH_SIZE], batch_vectors):
            embeddings[chunk.chunk_id] = vector
    return embeddings


def _get_embeddings_index() -> Tuple[List[CorpusChunk], Dict[str, List[float]]]:
    global _EMBEDDING_INDEX, _EMBEDDING_INDEX_FINGERPRINT, _EMBEDDING_INDEX_MODEL

    chunks, fingerprint = _get_corpus_chunks_and_fingerprint()
    model = _get_embedding_model()

    if (
        _EMBEDDING_INDEX is not None
        and _EMBEDDING_INDEX_FINGERPRINT == fingerprint
        and _EMBEDDING_INDEX_MODEL == model
    ):
        return chunks, _EMBEDDING_INDEX

    cache_path = _get_cache_path()
    embeddings: Optional[Dict[str, List[float]]] = None
    if cache_path:
        embeddings = _load_embedding_cache(cache_path, fingerprint, model)

    if embeddings is None:
        embeddings = _compute_embeddings(chunks)
        if cache_path:
            _save_embedding_cache(cache_path, fingerprint, model, embeddings)

    _EMBEDDING_INDEX = embeddings
    _EMBEDDING_INDEX_FINGERPRINT = fingerprint
    _EMBEDDING_INDEX_MODEL = model
    return chunks, embeddings


def _score_chunks_by_embedding(
    query_embedding: List[float],
    query_tokens: List[str],
    chunks: List[CorpusChunk],
    embeddings: Dict[str, List[float]],
) -> List[Tuple[float, CorpusChunk]]:
    scored: List[Tuple[float, CorpusChunk]] = []
    for chunk in chunks:
        vector = embeddings.get(chunk.chunk_id)
        if not vector:
            continue
        score = _cosine_similarity_dense(query_embedding, vector)
        if score <= 0.0:
            continue
        tag_boost = 0.05 * float(_tag_overlap(query_tokens, chunk.tags))
        scored.append((score + tag_boost, chunk))
    return scored


def _rank_chunks_by_tfidf(
    query_text: str, query_tokens: List[str], chunks: List[CorpusChunk]
) -> List[Tuple[float, CorpusChunk]]:
    if not chunks:
        return []
    texts = [chunk.text for chunk in chunks]
    vectors = _tfidf_vectors([query_text] + texts)
    query_vector = vectors[0]
    scored: List[Tuple[float, CorpusChunk]] = []
    for chunk, vector in zip(chunks, vectors[1:]):
        score = _cosine_similarity_sparse(query_vector, vector)
        if score <= 0.0:
            continue
        tag_boost = 0.05 * float(_tag_overlap(query_tokens, chunk.tags))
        scored.append((score + tag_boost, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _retrieve_with_tfidf(query_text: str, query_tokens: List[str]) -> List[Reference]:
    chunks, _ = _get_corpus_chunks_and_fingerprint()
    if not chunks:
        return []

    scored = _rank_chunks_by_tfidf(query_text, query_tokens, chunks)
    if not scored:
        fallback: List[Tuple[int, CorpusChunk]] = []
        for chunk in chunks:
            overlap = _tag_overlap(query_tokens, chunk.tags)
            if overlap > 0:
                fallback.append((overlap, chunk))
        if not fallback:
            return []
        fallback.sort(key=lambda item: item[0], reverse=True)
        selected = [chunk for _, chunk in fallback[:MAX_REFERENCES]]
    else:
        selected = [chunk for _, chunk in scored[:MAX_REFERENCES]]

    window = _get_context_window()
    by_source, index_map = _index_chunks_by_source(chunks)
    references: List[Reference] = []
    seen: set[Tuple[str, str]] = set()
    for chunk in selected:
        note = _expand_chunk_context(chunk, by_source, index_map, window)
        key = (chunk.source_title, note)
        if key in seen:
            continue
        seen.add(key)
        references.append(Reference(source=chunk.source_title, note=note))
    return references


def _retrieve_with_embeddings(query_text: str, query_tokens: List[str], mode: str) -> List[Reference]:
    try:
        chunks, embeddings = _get_embeddings_index()
    except Exception as exc:
        logger.warning("Embedding index unavailable: %s", exc)
        return []

    try:
        query_embedding = _embed_texts([query_text])[0]
    except Exception as exc:
        logger.warning("Embedding query failed: %s", exc)
        return []

    scored = _score_chunks_by_embedding(query_embedding, query_tokens, chunks, embeddings)
    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] < _get_embedding_threshold():
        return []

    top_k = min(_get_embedding_top_k(), len(scored))
    candidates = [chunk for _, chunk in scored[:top_k]]

    if mode == "embedding":
        selected = candidates[:MAX_REFERENCES]
        window = _get_context_window()
        by_source, index_map = _index_chunks_by_source(chunks)
        references: List[Reference] = []
        seen: set[Tuple[str, str]] = set()
        for chunk in selected:
            note = _expand_chunk_context(chunk, by_source, index_map, window)
            key = (chunk.source_title, note)
            if key in seen:
                continue
            seen.add(key)
            references.append(Reference(source=chunk.source_title, note=note))
        return references

    reranked = _rank_chunks_by_tfidf(query_text, query_tokens, candidates)
    if reranked:
        selected = [chunk for _, chunk in reranked[:MAX_REFERENCES]]
    else:
        selected = candidates[:MAX_REFERENCES]

    window = _get_context_window()
    by_source, index_map = _index_chunks_by_source(chunks)
    references: List[Reference] = []
    seen: set[Tuple[str, str]] = set()
    for chunk in selected:
        note = _expand_chunk_context(chunk, by_source, index_map, window)
        key = (chunk.source_title, note)
        if key in seen:
            continue
        seen.add(key)
        references.append(Reference(source=chunk.source_title, note=note))
    return references


def retrieve_evidence(request: RetrievalRequest) -> List[Reference]:
    """선택적 검색 계층. 참고 자료를 반환."""
    chunks, _ = _get_corpus_chunks_and_fingerprint()
    if not chunks:
        return []

    query_text = " ".join(
        [
            request.risk_stage,
            request.conversation_type,
            *request.signals,
            *request.query_terms,
            *request.matched_phrases,
        ]
    ).strip()
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    mode = _get_retriever_mode()
    if mode in {"embedding", "hybrid"}:
        references = _retrieve_with_embeddings(query_text, query_tokens, mode)
        if references:
            return references
        logger.info("Embedding retrieval fallback to TF-IDF")

    return _retrieve_with_tfidf(query_text, query_tokens)
