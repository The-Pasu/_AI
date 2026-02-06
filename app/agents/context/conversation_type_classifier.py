import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from app.core.logging import get_logger
from app.utils.text_patterns import CONVERSATION_TYPE_RULES
from app.utils.text_utils import normalize_text

logger = get_logger(__name__)


load_dotenv()


ALLOWED_CONTEXT_TYPES = ["구직", "중고거래", "재테크", "부업"]
EMBEDDING_MODEL_ENV = "OPENAI_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
PROTOTYPES_PATH = (
    Path(__file__).resolve().parents[1] / "analyzer" / "embedding_prototypes.json"
)

_EMBEDDING_CLIENT: Optional[OpenAI] = None
_PROTOTYPE_CENTROIDS: Optional[Dict[str, List[float]]] = None
_DEFAULT_CATEGORY: Optional[str] = None


def _get_client() -> OpenAI:
    global _EMBEDDING_CLIENT
    if _EMBEDDING_CLIENT is None:
        _EMBEDDING_CLIENT = OpenAI()
    return _EMBEDDING_CLIENT


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _embed_texts(texts: List[str]) -> List[List[float]]:
    client = _get_client()
    model = os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)
    response = client.embeddings.create(model=model, input=texts)
    data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in data]


def _load_prototypes() -> Tuple[Dict[str, List[str]], str]:
    with PROTOTYPES_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = payload.get("categories", [])
    prototype_map: Dict[str, List[str]] = {}
    default_category = None
    for item in categories:
        name = str(item.get("name", "")).strip()
        samples = [str(sample).strip() for sample in item.get("samples", []) if str(sample).strip()]
        if not name or not samples:
            continue
        prototype_map[name] = samples
        if default_category is None:
            default_category = name
    return prototype_map, default_category or ALLOWED_CONTEXT_TYPES[0]


def _get_prototype_centroids() -> Tuple[Dict[str, List[float]], str]:
    global _PROTOTYPE_CENTROIDS, _DEFAULT_CATEGORY
    if _PROTOTYPE_CENTROIDS is not None and _DEFAULT_CATEGORY is not None:
        return _PROTOTYPE_CENTROIDS, _DEFAULT_CATEGORY

    prototype_map, default_category = _load_prototypes()
    if not prototype_map:
        raise ValueError("No embedding prototypes found.")

    all_samples: List[str] = []
    category_slices: List[Tuple[str, int, int]] = []
    cursor = 0
    for category, samples in prototype_map.items():
        all_samples.extend(samples)
        category_slices.append((category, cursor, cursor + len(samples)))
        cursor += len(samples)

    embeddings = _embed_texts(all_samples)
    centroids: Dict[str, List[float]] = {}
    for category, start, end in category_slices:
        vectors = embeddings[start:end]
        if not vectors:
            continue
        vector_len = len(vectors[0])
        mean_vector = [0.0] * vector_len
        for vector in vectors:
            for idx in range(vector_len):
                mean_vector[idx] += vector[idx]
        count = float(len(vectors))
        mean_vector = [value / count for value in mean_vector]
        centroids[category] = mean_vector

    _PROTOTYPE_CENTROIDS = centroids
    _DEFAULT_CATEGORY = default_category
    return centroids, default_category


def _build_embedding_input(conversation: List[str], max_chars: int = 4000) -> str:
    parts = [line.strip() for line in conversation if line and line.strip()]
    text = "\n".join(parts).strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # 최근 대화가 더 중요할 가능성이 높아 tail을 우선 사용
    return text[-max_chars:]


def _rule_based_classify(conversation: List[str]) -> Optional[str]:
    scores: Dict[str, int] = {key: 0 for key in CONVERSATION_TYPE_RULES.keys()}
    for message in conversation:
        normalized = normalize_text(message)
        for conversation_type, patterns in CONVERSATION_TYPE_RULES.items():
            if any(pattern.search(normalized) for pattern in patterns):
                scores[conversation_type] += 1

    if not scores:
        return None

    best_type = max(scores, key=scores.get)
    if scores[best_type] <= 0:
        return None
    if best_type not in ALLOWED_CONTEXT_TYPES:
        return None
    return best_type


def classify_conversation_type(conversation: List[str]) -> str:
    """대화 유형 분류. risk_stage에는 영향을 주지 않음."""
    try:
        centroids, default_category = _get_prototype_centroids()
    except Exception as exc:
        logger.exception("Failed to load embedding prototypes: %s", exc)
        fallback_type = _rule_based_classify(conversation)
        if fallback_type:
            return fallback_type
        return ALLOWED_CONTEXT_TYPES[0]

    text = _build_embedding_input(conversation)
    if not text:
        fallback_type = _rule_based_classify(conversation)
        if fallback_type:
            return fallback_type
        return default_category

    try:
        embedding = _embed_texts([text])[0]
        scores = {
            category: _cosine_similarity(embedding, centroid)
            for category, centroid in centroids.items()
        }
        if not scores:
            fallback_type = _rule_based_classify(conversation)
            if fallback_type:
                return fallback_type
            return default_category
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_category = sorted_scores[0][0]
        if best_category not in ALLOWED_CONTEXT_TYPES:
            fallback_type = _rule_based_classify(conversation)
            if fallback_type:
                return fallback_type
            return default_category
        return best_category
    except Exception as exc:
        logger.exception("Embedding classification failed: %s", exc)
        fallback_type = _rule_based_classify(conversation)
        if fallback_type:
            return fallback_type
        return default_category
