from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

from sqlalchemy import select

from app.agents.reporting.report_generator import ReportInputs, generate_report_summary
from app.db.models import AnalysisRun, UserMemory
from app.db.session import get_session


def _to_date(value: datetime | date) -> date:
    return value.date() if isinstance(value, datetime) else value


def _select_timeline_runs(
    runs: List[AnalysisRun], max_items: int = 5
) -> List[AnalysisRun]:
    if not runs:
        return []
    sorted_runs = sorted(runs, key=lambda item: item.created_at, reverse=True)
    return sorted_runs[:max_items]


def _build_timeline_summaries(runs: List[AnalysisRun]) -> List[str]:
    summaries: List[str] = []
    for run in runs:
        date_text = _to_date(run.created_at)
        summary = (run.summary or "").strip()
        risk_stage = run.risk_stage
        if summary:
            summaries.append(f"{date_text} [{risk_stage}] {summary}")
    return summaries


def _aggregate_stats(runs: List[AnalysisRun]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    type_counts = Counter()
    stage_counts = Counter()
    signal_counts = Counter()

    for run in runs:
        type_counts.update([run.conversation_type])
        stage_counts.update([run.risk_stage])
        for signal in run.detected_signals or []:
            signal_counts.update([signal])

    return dict(type_counts), dict(stage_counts), dict(signal_counts)


def build_report(uuid: str) -> Dict[str, Any]:
    with get_session() as session:
        runs = list(
            session.scalars(
                select(AnalysisRun)
                .where(AnalysisRun.uuid == uuid)
                .order_by(AnalysisRun.created_at.asc())
            )
        )
        memory = session.scalar(select(UserMemory).where(UserMemory.uuid == uuid))
        if not runs:
            return {
                "uuid": uuid,
                "period": {"from": "", "to": ""},
                "overview": "요청 기간 내 분석 기록이 없습니다.",
                "risk_trends": [],
                "top_signals": [],
                "timeline_highlights": [],
                "recommendations": [],
            }

        period_from = _to_date(runs[0].created_at)
        period_to = _to_date(runs[-1].created_at)
        type_counts, stage_counts, signal_counts = _aggregate_stats(runs)
        timeline_runs = _select_timeline_runs(runs)
        timeline_summaries = _build_timeline_summaries(timeline_runs)
        rolling_summary = memory.rolling_summary if memory else ""

        inputs = ReportInputs(
            uuid=uuid,
            period_from=period_from,
            period_to=period_to,
            rolling_summary=rolling_summary,
            type_counts=type_counts,
            stage_counts=stage_counts,
            signal_counts=signal_counts,
            timeline_summaries=timeline_summaries,
        )

        overview, recommendations = generate_report_summary(inputs)
        if not overview:
            overview = "요청 기간 내 위험 신호 요약을 생성할 수 없습니다."

        return {
            "uuid": uuid,
            "period": {"from": str(period_from), "to": str(period_to)},
            "overview": overview,
            "risk_trends": [
                {"type": key, "count": value}
                for key, value in sorted(type_counts.items(), key=lambda item: item[1], reverse=True)
            ],
            "top_signals": [
                {"signal": key, "count": value}
                for key, value in sorted(signal_counts.items(), key=lambda item: item[1], reverse=True)
            ],
            "timeline_highlights": [
                {
                    "date": str(_to_date(run.created_at)),
                    "summary": run.summary,
                    "risk_stage": run.risk_stage,
                }
                for run in timeline_runs
            ],
            "recommendations": recommendations,
        }
