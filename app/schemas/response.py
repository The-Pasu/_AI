from typing import List

from pydantic import BaseModel, Field


class RagReference(BaseModel):
    source: str
    summary: str


class RiskSignal(BaseModel):
    quote: str
    reason: str


class AnalyzeResponse(BaseModel):
    summary: str
    type: str
    risk_signals: List[RiskSignal] = Field(default_factory=list)
    additional_recommendations: List[str] = Field(default_factory=list)
    rag_references: List[RagReference] = Field(default_factory=list)


class ReportPeriod(BaseModel):
    from_date: str = Field(..., alias="from")
    to_date: str = Field(..., alias="to")


class TrendItem(BaseModel):
    type: str
    count: int


class SignalItem(BaseModel):
    signal: str
    count: int


class TimelineItem(BaseModel):
    date: str
    summary: str
    risk_stage: str


class ReportResponse(BaseModel):
    uuid: str
    period: ReportPeriod
    overview: str
    risk_trends: List[TrendItem] = Field(default_factory=list)
    top_signals: List[SignalItem] = Field(default_factory=list)
    timeline_highlights: List[TimelineItem] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
