from fastapi import APIRouter

from app.schemas.request import ReportRequest
from app.schemas.response import ReportResponse
from app.services.report_service import build_report

router = APIRouter()


@router.post("/report", response_model=ReportResponse)
def report(payload: ReportRequest) -> ReportResponse:
    result = build_report(payload.uuid)
    return ReportResponse(**result)
