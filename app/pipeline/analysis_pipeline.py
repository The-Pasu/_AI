from typing import Dict, List, Sequence

from app.agents.actions.safe_action_generator import generate_safe_actions
from app.agents.analyzer.conversation_analyzer import (
    analyze_conversation,
    extract_signal_phrases,
    signal_query_terms,
)
from app.agents.decision.decision_orchestrator import decide_risk_stage
from app.agents.explanation.rag.rag_provider import retrieve_evidence
from app.agents.explanation.rag.retrieval_contract import RetrievalRequest
from app.core.logging import get_logger
from app.pipeline.message_preprocessor import normalize_messages_with_ocr
from app.schemas.request import AnalyzeRequest, Message
from app.utils.text_patterns import resolve_risk_signals
from app.utils.text_utils import normalize_text

logger = get_logger(__name__)


def _build_conversation_excerpt(
    messages: Sequence[Message], matched_phrases: List[str], max_lines: int = 20
) -> List[str]:
    if not messages:
        return []

    phrases = [normalize_text(phrase) for phrase in matched_phrases if phrase.strip()]
    lines = [f"{message.sender}: {message.content}" for message in messages]
    normalized_contents = [normalize_text(message.content) for message in messages]

    selected_indices: List[int] = []
    if phrases:
        for idx, normalized in enumerate(normalized_contents):
            if any(phrase in normalized for phrase in phrases):
                selected_indices.append(idx)

    # 최근 대화를 우선해서 부족한 라인을 채움
    idx = len(lines) - 1
    while len(selected_indices) < max_lines and idx >= 0:
        if idx not in selected_indices:
            selected_indices.append(idx)
        idx -= 1

    selected_indices = sorted(selected_indices)
    return [lines[idx] for idx in selected_indices]


def run_analysis_pipeline(payload: AnalyzeRequest) -> Dict[str, object]:
    conversation = normalize_messages_with_ocr(payload.messages)
    other_contents = [
        message.content
        for message in conversation
        if message.sender.strip().upper() == "OTHER" and message.content.strip()
    ]
    logger.info(
        "Pipeline start: %d turns (other=%d)",
        len(conversation),
        len(other_contents),
    )

    # 1. 입력에서 대화 유형 수신 (유형별 신호 범위 결정을 위함)
    conversation_type = payload.type
    logger.info("Step 1 conversation_type(from input): %s", conversation_type)

    # 2. 규칙 기반 신호 추출 (유형 기반 + 공통 신호) (위험 신호 후보 추출)
    allowed_signals = resolve_risk_signals(conversation_type)
    rule_signals = analyze_conversation(other_contents, allowed_signals=allowed_signals)
    logger.info("Step 2 signals: %s", rule_signals)

    # 3. RAG 쿼리 보강 (근거 자료 확보를 위한 검색 품질 향상)
    signal_terms = signal_query_terms(rule_signals)
    matched_phrases = extract_signal_phrases(other_contents, allowed_signals=allowed_signals)
    logger.info("Step 3 query_terms=%s matched_phrases=%s", signal_terms, matched_phrases)

    # 4. 결정 오케스트레이터 (위험 단계 산출)
    risk_stage = decide_risk_stage(rule_signals)
    logger.info("Step 4 risk_stage: %s", risk_stage)

    # 5. RAG 검색 (근거 자료 확보)
    retrieval_request = RetrievalRequest(
        risk_stage=risk_stage,
        conversation_type=conversation_type,
        signals=rule_signals,
        query_terms=signal_terms,
        matched_phrases=matched_phrases,
    )
    references = retrieve_evidence(retrieval_request)
    logger.info("Step 5 references: %d", len(references))

    # 6. 안전 행동 생성 (LLM: references + 대화 발췌 사용) (최종 응답 생성)
    conversation_lines = _build_conversation_excerpt(
        conversation, matched_phrases, max_lines=20
    )
    safe_actions = generate_safe_actions(
        risk_stage, conversation_type, references, conversation_lines, payload.platform
    )
    logger.info("Step 6 safe_actions generated")

    rag_references = [
        {"source": reference.source, "summary": reference.note} for reference in references
    ]

    return {
        "summary": safe_actions["summary"],
        "type": conversation_type,
        "risk_signals": safe_actions["risk_signals"],
        "additional_recommendations": safe_actions["additional_recommendations"],
        "recommended_questions": safe_actions.get("recommended_questions", []),
        "rag_references": rag_references,
        "risk_stage": risk_stage,
        "rule_signals": rule_signals,
    }
