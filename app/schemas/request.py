from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    type: Literal["TEXT", "URL"] = Field(
        ..., description="Message type (TEXT or URL)"
    )
    content: str = Field(..., description="Message content")
    sender: str = Field(..., description="Message sender (e.g. ME, OTHER)")
    timestamp: datetime = Field(..., description="Message timestamp in ISO 8601")
    
    @field_validator("type", mode="before")
    @classmethod
    def uppercase_type(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v
    
    @field_validator("sender", mode="before")
    @classmethod
    def uppercase_sender(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v


class AnalyzeRequest(BaseModel):
    uuid: str = Field(..., description="Request UUID")
    messages: List[Message] = Field(..., description="Ordered message list")
    platform: Literal["INSTAGRAM", "TELEGRAM"] = Field(
        ..., description="Platform name (INSTAGRAM or TELEGRAM)"
    )
    type: Optional[Literal["구직", "중고거래", "재테크", "부업"]] = Field(
        None, description="Conversation type (구직, 중고거래, 재테크, 부업)"
    )
    
    @field_validator("platform", mode="before")
    @classmethod
    def uppercase_platform(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v


class ReportRequest(BaseModel):
    uuid: str = Field(..., description="User UUID")
