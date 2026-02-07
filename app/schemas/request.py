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
    platform: str = Field(..., description="Platform name (e.g. TELEGRAM)")
