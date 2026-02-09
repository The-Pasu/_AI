from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.logging import get_logger

logger = get_logger(__name__)

REPORT_MODEL_ENV = "OPENAI_REPORT_MODEL"
DEFAULT_REPORT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class ReportInputs:
    uuid: str
    period_from: date
    period_to: date
    rolling_summary: str
    type_counts: Dict[str, int]
    stage_counts: Dict[str, int]
    signal_counts: Dict[str, int]
    timeline_summaries: List[str]


def _get_report_model() -> str:
    return os.getenv(REPORT_MODEL_ENV, os.getenv("OPENAI_MODEL_ENV", DEFAULT_REPORT_MODEL))


def _format_counts(title: str, counts: Dict[str, int]) -> str:
    if not counts:
        return f"{title}: 없음"
    sorted_items = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    items = ", ".join(f"{key}({value})" for key, value in sorted_items)
    return f"{title}: {items}"


def _build_prompt_inputs(inputs: ReportInputs) -> Dict[str, str]:
    return {
        "uuid": inputs.uuid,
        "period": f"{inputs.period_from} ~ {inputs.period_to}",
        "rolling_summary": inputs.rolling_summary or "(요약 없음)",
        "type_counts": _format_counts("유형 분포", inputs.type_counts),
        "stage_counts": _format_counts("위험 단계 분포", inputs.stage_counts),
        "signal_counts": _format_counts("주요 신호", inputs.signal_counts),
        "timeline": "\n".join(inputs.timeline_summaries) or "(타임라인 요약 없음)",
    }


def _build_chain() -> Tuple[ChatPromptTemplate, ChatOpenAI, StrOutputParser]:
    prompt = ChatPromptTemplate.from_template(
        """
너는 사용자 대화 기록을 요약해 리포트를 작성하는 분석가다.
아래 입력을 참고해 JSON만 출력하라.

요구사항:
- overview는 1~2문장 요약, 존댓말, 확정 판단 금지
- recommendations는 2~3개, 존댓말, 실행 가능한 조치
- 사실로 단정하지 말고 '정황/가능성/주의' 표현을 사용

입력:
- UUID: {uuid}
- 기간: {period}
- 누적 요약: {rolling_summary}
- {type_counts}
- {stage_counts}
- {signal_counts}
- 타임라인 요약:
{timeline}

출력(JSON):
{{"overview":"...","recommendations":["...","..."]}}
"""
    )
    llm = ChatOpenAI(model=_get_report_model())
    parser = StrOutputParser()
    return prompt, llm, parser


def _parse_payload(text: str) -> Tuple[str, List[str]]:
    cleaned = text.strip()
    if not cleaned:
        return "", []
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "", []
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return "", []

    overview = str(payload.get("overview", "")).strip()
    recommendations = payload.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []
    recommendations = [str(item).strip() for item in recommendations if str(item).strip()]
    return overview, recommendations


def generate_report_summary(inputs: ReportInputs) -> Tuple[str, List[str]]:
    prompt, llm, parser = _build_chain()
    chain = prompt | llm | parser
    data = _build_prompt_inputs(inputs)

    try:
        text = chain.invoke(data)
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)
        return "", []

    overview, recommendations = _parse_payload(text)
    if not overview:
        return "", recommendations
    return overview, recommendations
