from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ModelVariant(BaseModel):
    resolution: int = Field(..., description="Image resolution (e.g. 320, 480, 640)")
    format: str = Field(..., description="Mobile format (onnx, ncnn)")
    path: str = Field(..., description="Relative path in weights/mobile/")
    size_bytes: int = Field(..., description="Size in bytes")
    sha256: str = Field(..., description="SHA-256 hash")
    status: str = Field(default="ready", description="ready, not_prepared")

class ModelItem(BaseModel):
    id: str = Field(..., description="Unique model identifier, e.g. yolo26s-nav-291_best")
    name: str = Field(..., description="Display name, e.g. yolo26s-nav-291 (best.pt - mAP 0.37)")
    category: str = Field(default="training_checkpoint", description="training_best, training_checkpoint, base_weight, exported_mobile")
    run_id: Optional[str] = Field(default=None, description="Training run identifier")
    epoch: Optional[int] = Field(default=None, description="Checkpoint epoch number")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Validation metrics (mAP, loss, etc.)")
    variant: str = Field(default="n", description="Model variant (n, s, m, l, x)")
    supported_resolutions: List[int] = Field(default_factory=lambda: [320, 384, 480, 512, 640])
    resolution: int = Field(default=480, description="Default or specific square resolution")
    format: str = Field(default="pt", description="Source format (pt, onnx, ncnn)")
    source_path: str = Field(default="", description="Path to checkpoint source file")
    file_size_mb: float = Field(..., description="File size in megabytes")
    version: str = Field(default="1.0", description="Model version string")
    sha256: str = Field(default="", description="SHA-256 integrity hash")
    download_url: str = Field(..., description="Endpoint URL to download mobile model")
    exported_resolutions: List[int] = Field(default_factory=list, description="Resolutions already pre-exported in weights cache")
    classes: int = Field(default=27, description="Number of obstacle classes")
    class_names: List[str] = Field(default_factory=list, description="Ordered class names list")
    task: str = Field(default="detection", description="Task type: detection, depth")
    cdn_url: Optional[str] = Field(default=None, description="Direct high-speed GitHub/Cloudflare CDN URL")
    cdn_urls: Optional[Dict[int, str]] = Field(default=None, description="Resolution-specific CDN URLs mapping")
    variants: List[ModelVariant] = Field(default_factory=list, description="Pre-prepared mobile variants")

class ModelRegistryResponse(BaseModel):
    models: List[ModelItem]

class ModelMetadataResponse(BaseModel):
    id: str
    name: str
    category: str
    task: str = "detection"
    cdn_url: Optional[str] = None
    cdn_urls: Optional[Dict[int, str]] = None
    run_id: Optional[str] = None
    variant: str
    resolution: int
    format: str
    file_size_mb: float
    sha256: str
    classes: int
    class_names: List[str]
    input_shape: List[int]
    input_tensor_name: str
    output_tensor_name: str
    output_shape: List[int]
    stride: int
    mean: List[float] = [0.0, 0.0, 0.0]
    std: List[float] = [255.0, 255.0, 255.0]
    distance_heuristic: Dict[str, Any]
    variants: List[ModelVariant] = Field(default_factory=list)

class ModelStatusResponse(BaseModel):
    model_id: str
    resolution: int
    format: str
    status: str  # "ready" or "not_prepared"
    size_bytes: Optional[int] = None
    size_mb: Optional[float] = None
    sha256: Optional[str] = None
    filename: Optional[str] = None
    download_url: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    models_available: int
    server_ip: Optional[str] = None
