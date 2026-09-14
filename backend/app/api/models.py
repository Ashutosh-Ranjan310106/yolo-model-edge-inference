from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app.schemas.model_meta import (
    ModelRegistryResponse,
    ModelMetadataResponse,
    ModelItem,
    ModelStatusResponse
)
from app.services.registry import registry_service

router = APIRouter(prefix="/models", tags=["models"])

@router.get("", response_model=ModelRegistryResponse)
async def list_models(task: Optional[str] = Query(None, description="Filter by task: detection or depth")):
    """List all available models, checkpoints, and their prepared mobile variants."""
    models = registry_service.scan_models()
    if task:
        models = [m for m in models if m.task.lower() == task.lower()]
    return ModelRegistryResponse(models=models)

@router.get("/{model_id}/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_id: str,
    resolution: int = Query(default=480, description="Input square resolution (e.g. 320, 480, 640)"),
    format: str = Query(default="onnx", description="Mobile model format (onnx, ncnn)")
):
    """
    Check whether a specific mobile model variant is prepared and ready for immediate download.
    Possible states: ready, not_prepared.
    """
    return registry_service.get_variant_status(model_id, resolution=resolution, format=format)

@router.get("/{model_id}", response_model=ModelItem)
async def get_model(model_id: str):
    """Get metadata summary for a specific model or checkpoint."""
    model = registry_service.find_model_item(model_id)
    if model:
        return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")

@router.get("/{model_id}/metadata", response_model=ModelMetadataResponse)
async def get_model_metadata(
    model_id: str,
    resolution: int = Query(default=480, description="Input square resolution (e.g. 320, 480, 640)")
):
    """Get complete tensor and preprocessing metadata for a specific model/resolution."""
    meta = registry_service.get_model_metadata(model_id, resolution=resolution)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Metadata for model '{model_id}' not found.")
    return meta

@router.get("/{model_id}/checksum")
async def get_model_checksum(
    model_id: str,
    resolution: int = Query(default=480, description="Input square resolution"),
    format: str = Query(default="onnx", description="Mobile model format (onnx, ncnn)")
):
    """Get SHA-256 integrity checksum for an already-prepared model."""
    model_path = registry_service.get_prepared_model_path(model_id, resolution=resolution, format=format)
    if not model_path or not model_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "mobile_model_not_prepared",
                "message": f"{format.upper()} model for '{model_id}' at resolution {resolution} is not prepared yet."
            }
        )
    checksum = registry_service.calculate_sha256(model_path)
    return {"id": model_id, "resolution": resolution, "format": format, "sha256": checksum}

from fastapi.responses import JSONResponse, FileResponse, RedirectResponse

@router.get("/{model_id}/download")
async def download_model(
    model_id: str,
    resolution: int = Query(default=480, description="Target input resolution: 320, 384, 480, 512, 640"),
    format: str = Query(default="onnx", description="Target mobile format: onnx or ncnn"),
    redirect_cdn: bool = Query(default=False, description="Redirect to GitHub/Cloudflare CDN if available")
):
    """
    Stream an ALREADY PREPARED mobile model binary immediately.
    ZERO model export, conversion, or CUDA loading occurs during this request.
    If redirect_cdn=True and model has a CDN URL, returns HTTP 307 Redirect.
    """
    if redirect_cdn:
        item = registry_service.find_model_item(model_id)
        if item and item.cdn_url:
            return RedirectResponse(url=item.cdn_url, status_code=307)

    model_path = registry_service.get_prepared_model_path(model_id, resolution=resolution, format=format)

    if not model_path or not model_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "mobile_model_not_prepared",
                "message": f"{format.upper()} model for resolution {resolution} is not prepared yet.",
                "hint": f"Run 'python scripts/prepare_model.py --model {model_id} --resolution {resolution} --format {format}' on the server to prepare it."
            }
        )

    file_size = model_path.stat().st_size
    media_type = "application/x-onnx" if format.lower() == "onnx" else "application/octet-stream"

    return FileResponse(
        path=str(model_path),
        filename=model_path.name,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'attachment; filename="{model_path.name}"',
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Model-ID": model_id,
            "X-Model-Resolution": str(resolution),
            "X-Model-Format": format
        }
    )
