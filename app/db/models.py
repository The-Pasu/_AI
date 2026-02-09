from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ConversationMessage(Base):
    __tablename__ = "conversation_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    sender: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    message_type: Mapped[str] = mapped_column(String(16))
    platform: Mapped[str] = mapped_column(String(32))


class AnalysisRun(Base):
    __tablename__ = "analysis_run"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    conversation_type: Mapped[str] = mapped_column(String(32))
    risk_stage: Mapped[str] = mapped_column(String(16))
    detected_signals: Mapped[List[str]] = mapped_column(SQLiteJSON, default=list)
    summary: Mapped[str] = mapped_column(Text)
    rag_references: Mapped[List[Dict[str, Any]]] = mapped_column(SQLiteJSON, default=list)


class UserMemory(Base):
    __tablename__ = "user_memory"

    uuid: Mapped[str] = mapped_column(String(64), primary_key=True)
    rolling_summary: Mapped[str] = mapped_column(Text, default="")
    risk_stats: Mapped[Dict[str, Any]] = mapped_column(SQLiteJSON, default=dict)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
