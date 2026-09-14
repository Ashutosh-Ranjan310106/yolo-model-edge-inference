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

class NoCacheStaticFiles(StaticFiles):
    """
    Custom StaticFiles implementation that completely disables caching for frontend assets.
    Always returns HTTP 200 with fresh content and never returns 304 Not Modified.
    """
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        if "etag" in response.headers:
            del response.headers["etag"]
        if "last-modified" in response.headers:
            del response.headers["last-modified"]
        return response

# Disable frontend caching middleware - forces browser to always fetch fresh HTML, CSS, JS
@app.middleware("http")
async def disable_frontend_cache_middleware(request, call_next):
    if request.url.path.startswith("/client"):
        # Strip conditional request headers so backend and StaticFiles never return 304
        filtered_headers = [
            (name, val) for name, val in request.scope.get("headers", [])
            if name.lower() not in (b"if-none-match", b"if-modified-since")
        ]
        request.scope["headers"] = filtered_headers

    response = await call_next(request)

    if request.url.path.startswith("/client"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Surrogate-Control"] = "no-store"
        if "etag" in response.headers:
            del response.headers["etag"]
        if "last-modified" in response.headers:
            del response.headers["last-modified"]

    return response

# Include API & WebSocket Routers
app.include_router(health.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(inference_ws.router)

# Static frontend mount with zero caching
if settings.FRONTEND_DIR.exists():
    app.mount("/client", NoCacheStaticFiles(directory=str(settings.FRONTEND_DIR), html=True), name="frontend")

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
