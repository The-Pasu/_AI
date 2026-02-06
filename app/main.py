from fastapi import FastAPI

from app.api.analyze import router as analyze_router
from app.core.config import API_PREFIX, APP_NAME


def create_app() -> FastAPI:
    app = FastAPI(title=APP_NAME)
    app.include_router(analyze_router, prefix=API_PREFIX)
    return app


app = create_app()
