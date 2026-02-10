from fastapi import APIRouter

from app.core.logging import get_logger
from app.pipeline.analysis_pipeline import run_analysis_pipeline
from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalyzeResponse
from app.services.report_store import store_analysis_result

router = APIRouter()
logger = get_logger(__name__)


@router.post("/analyze", response_model=AnalyzeResponse, response_model_by_alias=True)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    result = run_analysis_pipeline(payload)
    try:
        store_analysis_result(payload, result)
    except Exception as exc:
        logger.warning("Failed to persist analysis result: %s", exc)
    return AnalyzeResponse(**result)
