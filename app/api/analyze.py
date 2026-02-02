from fastapi import APIRouter

from app.pipeline.analysis_pipeline import run_analysis_pipeline
from app.schemas.request import AnalyzeRequest
from app.schemas.response import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    result = run_analysis_pipeline(payload)
    return AnalyzeResponse(**result)
