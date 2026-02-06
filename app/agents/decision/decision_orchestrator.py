from typing import List, Literal


RiskStage = Literal["normal", "suspicious", "critical"]


def decide_risk_stage(signals: List[str]) -> RiskStage:
    """결정론적 판단 로직. LLM 사용 없음."""
    signal_set = set(signals)
    has_money_request = "money_request" in signal_set
    has_urgency = "urgency" in signal_set
    if has_money_request and has_urgency:
        return "critical"
    if has_money_request or has_urgency or signal_set:
        return "suspicious"
    return "normal"
