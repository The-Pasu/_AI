import json
import os
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from app.core.logging import get_logger
from app.agents.explanation.rag.retrieval_contract import Reference


load_dotenv()

OPENAI_MODEL_ENV = "OPENAI_MODEL_ENV"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"

logger = get_logger(__name__)


def _fallback_safe_actions(
    risk_stage: str, references: List[Reference]
) -> Dict[str, object]:
    if references:
        note = references[0].note.strip()
        summary = f"참고 자료에 따르면 {note}"
    elif risk_stage == "critical":
        summary = "진행 전 확인이 필요하다는 신호가 관찰되었습니다."
    elif risk_stage == "suspicious":
        summary = "추가 확인이 필요해 보이는 상황입니다."
    else:
        summary = "현재로서는 명확한 위험 징후가 보이지 않습니다."

    return {
        "summary": summary,
        "risk_signals": [],
        "additional_recommendations": [],
        "rag_references": [],
    }


def _parse_json_payload(text: str) -> Optional[Dict[str, object]]:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None


def _call_openai_safe_actions(
    risk_stage: str,
    conversation_type: str,
    references: List[Reference],
    conversation_lines: List[str],
) -> Optional[Dict[str, object]]:
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set; using fallback safe actions.")
        return None

    model = os.getenv(OPENAI_MODEL_ENV, DEFAULT_OPENAI_MODEL)

    reference_text = (
        "; ".join(f"{ref.source}: {ref.note}" for ref in references)
        if references
        else "없음"
    )

    def _conversation_excerpt(lines: List[str], max_lines: int = 20) -> str:
        if len(lines) <= max_lines:
            return "\n".join(lines)
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines - len(head)) :]
        return "\n".join(head + ["..."] + tail)

    conversation_excerpt = _conversation_excerpt(conversation_lines)

    instructions = (
        "너는 위험 단계에 맞는 요약, 위험 신호, 추가 권고를 만드는 어시스턴트다. "
        "반드시 JSON으로만 출력해라. "
        "summary는 1문장 요약이며, 참고 자료가 있으면 반드시 참고 자료를 근거로 작성한다. "
        "summary는 반드시 존댓말(입니다/합니다)로 작성한다. "
        "risk_signals는 대화에서 위험하다고 판단되는 내용을 인용하고, 왜 위험한지 짧게 설명한다. "
        "risk_signals의 각 항목은 quote(대화에서 그대로 인용)와 reason(왜 위험한지)로 구성한다. "
        "quote는 대화 발췌에 실제로 등장하는 문장/구절이어야 한다. "
        "additional_recommendations는 사용자가 취할 추가 확인/보호 조치를 2~4개로 제시한다. "
        "reason과 additional_recommendations는 반드시 존댓말(입니다/합니다/하세요)로 작성한다. "
        "대화 발췌 내용을 참고해 구체화해라. "
        "참고 자료에 없는 사실은 만들지 마라. "
        "중립적이고 설명적인 톤을 유지하고, 판단을 확정하지 마라. "
        "아래 형식을 참고해라.\n"
        "예시 입력 요약:\n"
        "- 위험 단계: suspicious\n"
        "- 대화 발췌: OTHER: \"등록비 5만원만 먼저 입금해 주세요.\"\n"
        "예시 출력(JSON):\n"
        "{"
        "\"summary\":\"입사 전 비용을 요구하는 정황이 있어 주의가 필요합니다.\","
        "\"risk_signals\":["
        "{\"quote\":\"등록비 5만원만 먼저 입금해 주세요.\","
        "\"reason\":\"입사 전 비용을 요구하는 방식은 사기 가능성이 있어 주의가 필요합니다.\"}"
        "],"
        "\"additional_recommendations\":["
        "\"공식 채용 공고와 회사 연락처로 사실 여부를 확인하세요.\","
        "\"입금 요청은 보류하고 서면 안내를 요청하세요.\""
        "]"
        "}"
        "\n"
        "예시 입력 요약:\n"
        "- 위험 단계: critical\n"
        "- 대화 발췌: OTHER: \"오늘 안에만 가능해요. 선입금 부탁드립니다.\" OTHER: \"안전결제 링크로 결제하세요.\"\n"
        "예시 출력(JSON):\n"
        "{"
        "\"summary\":\"중고거래에서 선입금과 링크 결제를 요구하는 정황이 있어 각별한 주의가 필요합니다.\","
        "\"risk_signals\":["
        "{\"quote\":\"오늘 안에만 가능해요. 선입금 부탁드립니다.\","
        "\"reason\":\"선입금을 요구하는 방식은 사기 가능성이 높아 주의가 필요합니다.\"},"
        "{\"quote\":\"안전결제 링크로 결제하세요.\","
        "\"reason\":\"외부 결제 링크 유도는 피싱 가능성이 있어 주의가 필요합니다.\"}"
        "],"
        "\"additional_recommendations\":["
        "\"플랫폼 내 안전결제 기능을 사용하고 외부 링크 결제는 피하세요.\","
        "\"직거래 또는 대면 확인이 가능한 방식으로 거래하세요.\""
        "]"
        "}"
        "\n"
        "예시 입력 요약:\n"
        "- 위험 단계: suspicious\n"
        "- 대화 발췌: OTHER: \"원금 보장 상품입니다. 오늘 안에만 가입 가능합니다.\"\n"
        "예시 출력(JSON):\n"
        "{"
        "\"summary\":\"원금 보장과 기한 제한을 강조하는 투자 제안은 주의가 필요합니다.\","
        "\"risk_signals\":["
        "{\"quote\":\"원금 보장 상품입니다. 오늘 안에만 가입 가능합니다.\","
        "\"reason\":\"원금 보장과 즉시 가입을 요구하는 표현은 사기 위험 신호일 수 있습니다.\"}"
        "],"
        "\"additional_recommendations\":["
        "\"공식 금융기관 등록 여부와 공시 정보를 먼저 확인하세요.\","
        "\"즉시 가입 요구는 보류하고 충분히 검토하세요.\""
        "]"
        "}"
    )
    prompt = (
        f"위험 단계: {risk_stage}\n"
        f"대화 유형: {conversation_type}\n"
        f"참고 자료: {reference_text}\n"
        f"대화 발췌:\n{conversation_excerpt}\n"
        "출력은 JSON 형식이어야 한다."
    )

    try:
        client = OpenAI()
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=400,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "safe_actions",
                    "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "risk_signals": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "quote": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["quote", "reason"],
                            },
                            "minItems": 0,
                            "maxItems": 6,
                        },
                        "additional_recommendations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 4,
                        },
                    },
                    "required": [
                        "summary",
                        "risk_signals",
                        "additional_recommendations",
                    ],
                },
                "strict": True,
            }
        },
        )
        text = response.output_text.strip()
        if not text:
            logger.warning("OpenAI safe actions returned empty output; using fallback.")
            return None
        payload = _parse_json_payload(text)
        if not payload:
            logger.warning("OpenAI safe actions returned invalid JSON; using fallback.")
            return None
        summary = str(payload.get("summary", "")).strip()
        risk_signals = payload.get("risk_signals", [])
        recommendations = payload.get("additional_recommendations", [])
        if not summary or not isinstance(risk_signals, list) or not isinstance(
            recommendations, list
        ):
            logger.warning("OpenAI safe actions missing fields; using fallback.")
            return None
        cleaned_signals = []
        for item in risk_signals:
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not quote or not reason:
                continue
            cleaned_signals.append({"quote": quote, "reason": reason})
        cleaned_recommendations = [
            str(item).strip() for item in recommendations if str(item).strip()
        ]
        if len(cleaned_recommendations) < 2:
            logger.warning("OpenAI safe actions recommendations invalid; using fallback.")
            return None
        return {
            "summary": summary,
            "risk_signals": cleaned_signals,
            "additional_recommendations": cleaned_recommendations,
            "rag_references": [],
        }
    except Exception as exc:
        logger.exception("OpenAI safe actions failed: %s", exc)
        return None


def generate_safe_actions(
    risk_stage: str,
    conversation_type: str,
    references: List[Reference],
    conversation_lines: List[str],
) -> Dict[str, object]:
    """요약, 위험 신호, 추가 권고를 생성."""
    llm_result = _call_openai_safe_actions(
        risk_stage, conversation_type, references, conversation_lines
    )
    if llm_result:
        return llm_result

    return _fallback_safe_actions(risk_stage, references)
