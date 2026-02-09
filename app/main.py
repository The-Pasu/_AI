from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.analyze import router as analyze_router
from app.api.report import router as report_router
from app.core.config import API_PREFIX, APP_NAME
from app.db.session import init_db


def _validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.include_router(analyze_router, prefix=API_PREFIX)
    app.include_router(report_router, prefix=API_PREFIX)

    @app.on_event("startup")
    def _startup() -> None:
        init_db()

    return app


app = create_app()
