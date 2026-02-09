from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    type: Literal["TEXT", "URL"] = Field(
        ..., description="Message type (TEXT or URL)"
    )
    content: str = Field(..., description="Message content")
    sender: str = Field(..., description="Message sender (e.g. ME, OTHER)")
    timestamp: datetime = Field(..., description="Message timestamp in ISO 8601")


class AnalyzeRequest(BaseModel):
    uuid: str = Field(..., description="Request UUID")
    messages: List[Message] = Field(..., description="Ordered message list")
    platform: Literal["INSTAGRAM", "TELEGRAM"] = Field(
        ..., description="Platform name (INSTAGRAM or TELEGRAM)"
    )
    type: Literal["구직", "중고거래", "재테크", "부업"] = Field(
        ..., description="Conversation type (구직, 중고거래, 재테크, 부업)"
    )


class ReportRequest(BaseModel):
    uuid: str = Field(..., description="User UUID")
