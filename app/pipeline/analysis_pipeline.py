from typing import Dict

from app.agents.actions.safe_action_generator import generate_safe_actions
from app.agents.analyzer.conversation_analyzer import (
    analyze_conversation,
    extract_signal_phrases,
    signal_query_terms,
)
from app.agents.context.conversation_type_classifier import classify_conversation_type
from app.agents.decision.decision_orchestrator import decide_risk_stage
from app.agents.explanation.rag.rag_provider import retrieve_evidence
from app.agents.explanation.rag.retrieval_contract import RetrievalRequest
from app.core.logging import get_logger
from app.schemas.request import AnalyzeRequest
from app.utils.text_patterns import resolve_risk_signals

logger = get_logger(__name__)


def run_analysis_pipeline(payload: AnalyzeRequest) -> Dict[str, object]:
    conversation = payload.messages
    contents = [message.content for message in conversation]
    other_contents = [
        message.content
        for message in conversation
        if message.sender.strip().upper() == "OTHER"
    ]
    logger.info("Pipeline start: %d turns (other=%d)", len(conversation), len(other_contents))

    # 1. 대화 유형 분류 (임베딩 + fallback) (유형별 신호 범위 결정을 위함)
    conversation_type = classify_conversation_type(contents)
    logger.info("Step 1 conversation_type: %s", conversation_type)

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
    conversation_lines = [f"{message.sender}: {message.content}" for message in conversation]
    safe_actions = generate_safe_actions(
        risk_stage, conversation_type, references, conversation_lines
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
        "rag_references": rag_references,
    }
