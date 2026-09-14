import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.config.settings import settings
from app.api import health, models
from app.websocket import inference_ws
from app.services.registry import registry_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Dedicated Model Registry and Distribution Server for ROD Mobile Edge Inference."
)

# CORS configuration - Allows Android phone / browser access from LAN & tunnels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Content-Length", "Accept-Ranges", "X-Model-ID", "X-Model-Resolution", "X-Model-Format", "*"],
)

# Development No-Cache Middleware - ensures mobile phones and browsers always fetch fresh JS/CSS (bypasses 304)
@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/client"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Include API & WebSocket Routers
app.include_router(health.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(inference_ws.router)

# Static frontend mount if frontend exists
if settings.FRONTEND_DIR.exists():
    app.mount("/client", StaticFiles(directory=str(settings.FRONTEND_DIR), html=True), name="frontend")

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to /client/ for instant mobile browser testing."""
    if settings.FRONTEND_DIR.exists():
        return RedirectResponse(url="/client/")
    return {"message": settings.PROJECT_NAME, "docs": "/docs", "models": "/api/models"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi import Response
    return Response(content=b"", media_type="image/x-icon")

@app.on_event("startup")
async def startup_banner():
    local_ip = registry_service.get_local_ip()
    model_count = len(registry_service.scan_models())
    print("\n" + "=" * 64)
    print(f" {settings.PROJECT_NAME} v{settings.VERSION}")
    print("=" * 64)
    print(f" * Server running on: http://localhost:{settings.PORT}")
    print(f" * Mobile LAN URL   : http://{local_ip}:{settings.PORT}/client/")
    print(f" * API Docs         : http://{local_ip}:{settings.PORT}/docs")
    print(f" * Models Available : {model_count}")
    print("=" * 64 + "\n")

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
