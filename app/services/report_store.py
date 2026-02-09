from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List

from sqlalchemy import select

from app.db.models import AnalysisRun, ConversationMessage, UserMemory
from app.db.session import get_session
from app.schemas.request import AnalyzeRequest, Message


def _to_message_rows(
    uuid: str, platform: str, messages: Iterable[Message]
) -> List[ConversationMessage]:
    rows: List[ConversationMessage] = []
    for message in messages:
        rows.append(
            ConversationMessage(
                uuid=uuid,
                timestamp=message.timestamp,
                sender=message.sender,
                content=message.content,
                message_type=message.type,
                platform=platform,
            )
        )
    return rows


def _update_risk_stats(
    risk_stats: Dict[str, Any],
    conversation_type: str,
    risk_stage: str,
    detected_signals: List[str],
    last_seen_at: datetime,
) -> Dict[str, Any]:
    type_counts = dict(risk_stats.get("type_counts", {}))
    stage_counts = dict(risk_stats.get("stage_counts", {}))
    signal_counts = dict(risk_stats.get("signal_counts", {}))

    type_counts[conversation_type] = int(type_counts.get(conversation_type, 0)) + 1
    stage_counts[risk_stage] = int(stage_counts.get(risk_stage, 0)) + 1
    for signal in detected_signals:
        signal_counts[signal] = int(signal_counts.get(signal, 0)) + 1

    return {
        "type_counts": type_counts,
        "stage_counts": stage_counts,
        "signal_counts": signal_counts,
        "last_seen_at": last_seen_at.isoformat(),
    }


def store_analysis_result(
    payload: AnalyzeRequest,
    result: Dict[str, Any],
) -> None:
    uuid = payload.uuid
    platform = payload.platform
    conversation_type = payload.type
    risk_stage = str(result.get("risk_stage", "normal"))
    detected_signals = list(result.get("rule_signals", []))
    summary = str(result.get("summary", "")).strip()
    rag_references = list(result.get("rag_references", []))
    created_at = datetime.utcnow()

    message_rows = _to_message_rows(uuid, platform, payload.messages)

    with get_session() as session:
        if message_rows:
            session.add_all(message_rows)

        analysis_run = AnalysisRun(
            uuid=uuid,
            created_at=created_at,
            conversation_type=conversation_type,
            risk_stage=risk_stage,
            detected_signals=detected_signals,
            summary=summary,
            rag_references=rag_references,
        )
        session.add(analysis_run)

        memory = session.scalar(select(UserMemory).where(UserMemory.uuid == uuid))
        if memory is None:
            memory = UserMemory(
                uuid=uuid,
                rolling_summary=summary,
                risk_stats=_update_risk_stats({}, conversation_type, risk_stage, detected_signals, created_at),
                last_updated_at=created_at,
            )
            session.add(memory)
        else:
            rolling_summary = memory.rolling_summary or ""
            if summary:
                if rolling_summary:
                    rolling_summary = f"{rolling_summary}\n[{created_at.date()}] {summary}"
                else:
                    rolling_summary = f"[{created_at.date()}] {summary}"
            memory.rolling_summary = rolling_summary
            memory.risk_stats = _update_risk_stats(
                memory.risk_stats or {},
                conversation_type,
                risk_stage,
                detected_signals,
                created_at,
            )
            memory.last_updated_at = created_at
