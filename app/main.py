from fastapi import FastAPI

from app.api.diagnosis import router as diagnosis_router
from app.api.health import router as health_router
from app.api.manual import router as manual_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.include_router(health_router, prefix="/api")
    app.include_router(diagnosis_router, prefix="/api")
    app.include_router(manual_router, prefix="/api")
    return app


app = create_app()
