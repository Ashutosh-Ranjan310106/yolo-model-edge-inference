from fastapi import APIRouter
from app.config.settings import settings
from app.schemas.model_meta import HealthResponse
from app.services.registry import registry_service

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def get_health():
    models = registry_service.scan_models()
    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        models_available=len(models),
        server_ip=registry_service.get_local_ip()
    )
