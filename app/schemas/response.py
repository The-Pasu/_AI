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
