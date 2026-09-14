import hashlib
import json
import re
import socket
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.config.settings import settings
from app.schemas.model_meta import ModelItem, ModelMetadataResponse, ModelVariant

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# 46 Classes from navigation_dataset/local_dataset.yaml
LOCAL_46_CLASSES = [
    "Bike", "Building", "Car", "Person", "Stairs",
    "Traffic sign", "Electrical Pole", "Road", "Motorcycle", "Dustbin",
    "Dog", "Manhole", "Tree", "Guard rail", "Pedestrian crosswalk",
    "Truck", "Bus", "Bench", "Traffic Cone", "Fire hydrant",
    "Teraffic Barrel", "Plant Pot", "Electrical Box", "Chair",
    "Bicycle Rack", "Door", "Wall", "Bus station", "Barricade",
    "Table", "Desk", "Bookshelf/Storage", "Window", "Sink",
    "Toilet", "Sign_Board", "Elevator", "Train", "Cat",
    "Traffic Light", "Ramp", "Escalator", "Animal", "Computer",
    "Drawer", "TV"
]

# Strictly defined models: Converted Saved (ROD) and Official Baseline (COCO)
KNOWN_MODELS = [
    # --- Category: Converted Saved Models (Fine-tuned ROD Runs) ---
    {
        "id": "yolo26n-nav-run6_320x320",
        "name": "⚡ yolo26n-nav-run6 (Best - 46 Classes)",
        "category": "converted_saved",
        "variant": "n",
        "run_id": "yolo26n-nav-run6_320x320",
        "source_path": settings.TRAINING_RUNS_DIR / "yolo26n-nav-run6_320x320" / "weights" / "best.pt",
        "classes": LOCAL_46_CLASSES,
        "resolutions": [320, 480],
        "is_end2end": True,
        "cdn_urls": {
            320: "https://raw.githubusercontent.com/Aryan1q/DRISHTIX/master/navigation/mobile_distributable_models/yolo26n-nav-run6_320.onnx",
            480: "https://raw.githubusercontent.com/Aryan1q/DRISHTIX/master/navigation/mobile_distributable_models/yolo26n-nav-run6_480.onnx"
        }
    },
    {
        "id": "yolo26n-24k-nav-run1_best",
        "name": "⚡ yolo26n-24k-nav-run1 (Best - 24k Dataset)",
        "category": "converted_saved",
        "variant": "n",
        "run_id": "yolo26n-24k-nav-run1",
        "source_path": settings.TRAINING_RUNS_DIR / "yolo26n-24k-nav-run1" / "weights" / "best.pt",
        "classes": settings.ROD_CLASSES,
        "resolutions": [320, 480, 640],
        "is_end2end": False,
        "cdn_urls": {
            320: "https://raw.githubusercontent.com/Aryan1q/DRISHTIX/master/navigation/mobile_distributable_models/yolo26n-24k-nav-run1_best_320.onnx"
        }
    },
    {
        "id": "yolo26s-nav-291_best",
        "name": "⚡ yolo26s-nav-291 (Best - Run 291)",
        "category": "converted_saved",
        "variant": "s",
        "run_id": "yolo26s-nav-291",
        "source_path": settings.TRAINING_RUNS_DIR / "yolo26s-nav-291" / "weights" / "best.pt",
        "classes": settings.ROD_CLASSES,
        "resolutions": [320, 480]
    },
    {
        "id": "yolo26s-nav-408_best",
        "name": "⚡ yolo26s-nav-408 (Best - Run 408)",
        "category": "converted_saved",
        "variant": "s",
        "run_id": "yolo26s-nav-408",
        "source_path": settings.TRAINING_RUNS_DIR / "yolo26s-nav-408" / "weights" / "best.pt",
        "classes": settings.ROD_CLASSES,
        "resolutions": [320, 480]
    },
    {
        "id": "yolo26s-nav-728_best",
        "name": "⚡ yolo26s-nav-728 (Best - Run 728)",
        "category": "converted_saved",
        "variant": "s",
        "run_id": "yolo26s-nav-728",
        "source_path": settings.TRAINING_RUNS_DIR / "yolo26s-nav-728" / "weights" / "best.pt",
        "classes": settings.ROD_CLASSES,
        "resolutions": [320, 480]
    },
    {
        "id": "yolov26n_best",
        "name": "⚡ yolov26n_best (Best Baseline)",
        "category": "converted_saved",
        "variant": "n",
        "run_id": None,
        "source_path": settings.WEIGHTS_DIR / "yolov26n_best.pt",
        "classes": settings.ROD_CLASSES,
        "resolutions": [320, 480]
    },

    # --- Category: Official Baseline Models ---
    {
        "id": "yolo26n",
        "name": "📦 YOLO26 Nano (Official COCO)",
        "category": "official_model",
        "variant": "n",
        "run_id": None,
        "source_path": settings.NAVIGATION_DIR / "yolo26n.pt",
        "classes": COCO_CLASSES,
        "resolutions": [320, 480]
    },
    {
        "id": "yolo26s",
        "name": "📦 YOLO26 Small (Official COCO)",
        "category": "official_model",
        "variant": "s",
        "run_id": None,
        "source_path": settings.NAVIGATION_DIR / "yolo26s.pt",
        "classes": COCO_CLASSES,
        "resolutions": [320, 480]
    },
    {
        "id": "yolo26m",
        "name": "📦 YOLO26 Medium (Official COCO)",
        "category": "official_model",
        "variant": "m",
        "run_id": None,
        "source_path": settings.NAVIGATION_DIR / "yolo26m.pt",
        "classes": COCO_CLASSES,
        "resolutions": [320, 480]
    },
    {
        "id": "yolo11n",
        "name": "📦 YOLO11 Nano (Official COCO)",
        "category": "official_model",
        "variant": "n",
        "run_id": None,
        "source_path": settings.NAVIGATION_DIR / "yolo11n.pt",
        "classes": COCO_CLASSES,
        "resolutions": [320, 480]
    },
    {
        "id": "yolov8n",
        "name": "📦 YOLOv8 Nano (Official COCO)",
        "category": "official_model",
        "task": "detection",
        "variant": "n",
        "run_id": None,
        "source_path": settings.NAVIGATION_DIR / "yolov8n.pt",
        "classes": COCO_CLASSES,
        "resolutions": [320, 480]
    },

    # --- Category: Depth Estimation Models ---
    {
        "id": "depth_anything_v2_metric_small",
        "name": "📐 Depth Anything V2 Metric Small (Metric in Meters)",
        "category": "depth_model",
        "task": "depth",
        "variant": "s",
        "run_id": None,
        "source_path": settings.WEIGHTS_DIR / "depth" / "depth_anything_v2_metric_small.onnx",
        "classes": [],
        "resolutions": [518],
        "input_tensor_name": "pixel_values",
        "output_tensor_name": "predicted_depth",
        "output_shape": [1, 518, 518],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "is_metric": True
    },
    {
        "id": "depth_anything_v2_small_quantized",
        "name": "⚡ Depth Anything V2 Small (Mobile INT8 - 27MB)",
        "category": "depth_model",
        "task": "depth",
        "variant": "s",
        "run_id": None,
        "source_path": settings.WEIGHTS_DIR / "depth" / "depth_anything_v2_small_quantized.onnx",
        "classes": [],
        "resolutions": [518],
        "input_tensor_name": "pixel_values",
        "output_tensor_name": "predicted_depth",
        "output_shape": [1, 518, 518],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "is_metric": False,
        "cdn_url": "https://raw.githubusercontent.com/Aryan1q/DRISHTIX/master/navigation/mobile_distributable_models/depth_anything_v2_small_quantized.onnx"
    },
    {
        "id": "yolo26s_depth",
        "name": "📦 YOLO26 Small Depth (Official yolos-depth - 46MB)",
        "category": "depth_model",
        "task": "depth",
        "variant": "s",
        "run_id": None,
        "source_path": settings.WEIGHTS_DIR / "depth" / "yolo26s_depth.onnx",
        "classes": [],
        "resolutions": [768],
        "input_tensor_name": "images",
        "output_tensor_name": "depth",
        "output_shape": [1, 1, 768, 768],
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "is_metric": True,
        "cdn_url": "https://raw.githubusercontent.com/Aryan1q/DRISHTIX/master/navigation/mobile_distributable_models/yolo26s_depth.onnx"
    },
    {
        "id": "yolo26n_depth",
        "name": "📦 YOLO26 Nano Depth (Official YOLO Depth - 20MB)",
        "category": "depth_model",
        "task": "depth",
        "variant": "n",
        "run_id": None,
        "source_path": settings.WEIGHTS_DIR / "depth" / "yolo26n_depth.onnx",
        "classes": [],
        "resolutions": [768],
        "input_tensor_name": "images",
        "output_tensor_name": "depth",
        "output_shape": [1, 1, 768, 768],
        "mean": [0.0, 0.0, 0.0],
        "std": [255.0, 255.0, 255.0],
        "is_metric": True,
        "cdn_url": "https://raw.githubusercontent.com/Aryan1q/DRISHTIX/master/navigation/mobile_distributable_models/yolo26n_depth.onnx"
    }
]

class ModelRegistryService:
    def __init__(self, weights_dir: Optional[Path] = None):
        self.weights_dir = weights_dir or settings.WEIGHTS_DIR
        self.mobile_dir = self.weights_dir / "mobile"
        self.original_dir = self.weights_dir / "original"

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.mobile_dir.mkdir(parents=True, exist_ok=True)
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self._sha_cache: Dict[str, str] = {}

    @staticmethod
    def get_local_ip() -> str:
        """Get the primary local IPv4 address for mobile devices to connect over LAN."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file with in-memory caching."""
        if not file_path.exists():
            return ""
        cache_key = f"{file_path}_{file_path.stat().st_mtime}"
        if cache_key in self._sha_cache:
            return self._sha_cache[cache_key]

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        self._sha_cache[cache_key] = digest
        return digest

    def get_prepared_model_path(self, model_id: str, resolution: int = 480, format: str = "onnx") -> Optional[Path]:
        """
        Locates an ALREADY PREPARED mobile model binary.
        CRITICAL ARCHITECTURE REQUIREMENT:
        This function NEVER exports models or triggers long-running conversions!
        Returns Path if pre-exported file exists, otherwise None.
        """
        candidates = [
            # Direct files
            self.mobile_dir / f"{model_id}_{resolution}.{format}",
            self.mobile_dir / f"{model_id}.{format}",
            # Depth subdirectories
            self.mobile_dir / "depth" / f"{model_id}.{format}",
            self.mobile_dir / "depth" / f"{model_id}_{resolution}.{format}",
            self.weights_dir / "depth" / f"{model_id}.{format}",
            self.weights_dir / "depth" / f"{model_id}_{resolution}.{format}",
            # Standard aliases
            self.mobile_dir / f"official_{model_id}_{resolution}.{format}",
            self.mobile_dir / f"base_{model_id}_{resolution}.{format}",
            self.weights_dir / f"{model_id}_{resolution}.{format}",
            self.weights_dir / f"{model_id}.{format}",
            self.weights_dir / f"official_{model_id}_{resolution}.{format}",
            self.weights_dir / f"base_{model_id}_{resolution}.{format}",
        ]
        if "best" in model_id:
            stripped = model_id.replace("_best", "")
            candidates.extend([
                self.mobile_dir / f"{stripped}_{resolution}.{format}",
                self.weights_dir / f"{stripped}_{resolution}.{format}",
            ])
        else:
            candidates.extend([
                self.mobile_dir / f"{model_id}_best_{resolution}.{format}",
                self.weights_dir / f"{model_id}_best_{resolution}.{format}",
            ])

        # Specific alias for run6_320x320 -> run6
        if "320x320" in model_id:
            base_run = model_id.replace("_320x320", "")
            candidates.extend([
                self.mobile_dir / f"{base_run}_{resolution}.{format}",
                self.weights_dir / f"{base_run}_{resolution}.{format}",
            ])

        # Aliases for yolos-depth -> yolo26s_depth
        if "depth" in model_id:
            norm_id = model_id.replace("-", "_")
            candidates.extend([
                self.mobile_dir / "depth" / f"{norm_id}.{format}",
                self.weights_dir / "depth" / f"{norm_id}.{format}",
                self.mobile_dir / "depth" / f"yolo26s_depth.{format}",
            ])

        for cand in candidates:
            if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                return cand.resolve()

        return None

    def get_variant_status(self, model_id: str, resolution: int = 480, format: str = "onnx") -> Dict[str, Any]:
        """Return readiness status for a specific model variant."""
        path = self.get_prepared_model_path(model_id, resolution, format)
        if path and path.exists():
            size_bytes = path.stat().st_size
            return {
                "model_id": model_id,
                "resolution": resolution,
                "format": format,
                "status": "ready",
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "sha256": self.calculate_sha256(path),
                "filename": path.name,
                "download_url": f"/api/models/{model_id}/download?resolution={resolution}&format={format}"
            }
        else:
            return {
                "model_id": model_id,
                "resolution": resolution,
                "format": format,
                "status": "not_prepared",
                "message": f"{format.upper()} model for '{model_id}' at resolution {resolution} is not prepared."
            }

    def _get_variants_for_model(self, model_id: str, candidate_resolutions: List[int]) -> List[ModelVariant]:
        """Collect all prepared variants for a model."""
        variants: List[ModelVariant] = []
        for res in candidate_resolutions:
            cand = self.get_prepared_model_path(model_id, res, "onnx")
            if cand and cand.exists():
                variants.append(ModelVariant(
                    resolution=res,
                    format="onnx",
                    path=f"weights/mobile/{cand.name}",
                    size_bytes=cand.stat().st_size,
                    sha256=self.calculate_sha256(cand),
                    status="ready"
                ))
        variants.sort(key=lambda x: x.resolution)
        return variants

    def scan_models(self) -> List[ModelItem]:
        """
        Returns ONLY pre-converted saved models, official baseline models, and depth models.
        Zero un-exported checkpoints, zero on-demand compilation.
        """
        models: List[ModelItem] = []

        for cfg in KNOWN_MODELS:
            model_id = cfg["id"]
            cand_res = cfg.get("resolutions", [320, 480])
            variants = self._get_variants_for_model(model_id, cand_res)

            # Only include models that actually have prepared variants ready
            if not variants:
                continue

            exported_res = [v.resolution for v in variants if v.status == "ready"]
            first_size_mb = round(variants[0].size_bytes / (1024 * 1024), 2)
            source_p = cfg["source_path"]
            source_str = str(source_p.resolve()) if isinstance(source_p, Path) and source_p.exists() else str(source_p)

            item = ModelItem(
                id=model_id,
                name=cfg["name"],
                category=cfg["category"],
                run_id=cfg["run_id"],
                epoch=None,
                metrics=None,
                variant=cfg["variant"],
                supported_resolutions=exported_res,
                resolution=exported_res[0],
                format="onnx",
                source_path=source_str,
                file_size_mb=first_size_mb,
                version="1.0",
                sha256=variants[0].sha256,
                download_url=f"/api/models/{model_id}/download",
                exported_resolutions=exported_res,
                classes=len(cfg.get("classes", [])),
                class_names=cfg.get("classes", []),
                task=cfg.get("task", "detection"),
                cdn_url=cfg.get("cdn_urls", {}).get(exported_res[0], cfg.get("cdn_url")),
                cdn_urls=cfg.get("cdn_urls"),
                variants=variants
            )
            models.append(item)

        # Sort: converted_saved first, then official_model, then depth_model
        category_order = {"converted_saved": 0, "official_model": 1, "depth_model": 2}
        models.sort(key=lambda m: (category_order.get(m.category, 99), m.name))
        return models

    def find_model_item(self, model_id: str) -> Optional[ModelItem]:
        """Find a model item by its ID."""
        for m in self.scan_models():
            if m.id == model_id:
                return m
            # Handle aliases e.g. yolov26n vs yolov26n_best
            if model_id.endswith("_best") and m.id == model_id.replace("_best", ""):
                return m
            if not model_id.endswith("_best") and m.id == f"{model_id}_best":
                return m
        return None

    def get_model_metadata(self, model_id: str, resolution: int = 480) -> Optional[ModelMetadataResponse]:
        """Return comprehensive metadata for on-device inference setup."""
        model_item = self.find_model_item(model_id)
        if not model_item:
            for res in settings.SUPPORTED_RESOLUTIONS:
                if model_id.endswith(f"_{res}"):
                    base_id = model_id[:-len(f"_{res}")]
                    model_item = self.find_model_item(base_id)
                    resolution = res
                    break

        if not model_item:
            return None

        if model_item.supported_resolutions and resolution not in model_item.supported_resolutions:
            resolution = model_item.supported_resolutions[0]

        cfg = next((c for c in KNOWN_MODELS if c["id"] == model_item.id), {})
        is_depth = model_item.task == "depth"

        prepared_path = self.get_prepared_model_path(model_id, resolution, "onnx")
        if prepared_path and prepared_path.exists():
            sha256 = self.calculate_sha256(prepared_path)
            file_size_mb = round(prepared_path.stat().st_size / (1024 * 1024), 2)
        else:
            sha256 = model_item.sha256
            file_size_mb = model_item.file_size_mb

        cdn_urls = cfg.get("cdn_urls")
        cdn_url = cfg.get("cdn_url")
        if cdn_urls and resolution in cdn_urls:
            cdn_url = cdn_urls[resolution]
        elif cdn_urls:
            cdn_url = next(iter(cdn_urls.values()), cdn_url)

        if is_depth:
            input_tensor_name = cfg.get("input_tensor_name", "pixel_values")
            output_tensor_name = cfg.get("output_tensor_name", "predicted_depth")
            out_shape = cfg.get("output_shape", [1, resolution, resolution])
            mean_vals = cfg.get("mean", [0.485, 0.456, 0.406])
            std_vals = cfg.get("std", [0.229, 0.224, 0.225])
            classes_count = 0
            class_names = []
            stride_val = 14
            dist_heuristic = {
                "strategy": "dense_depth_map",
                "is_metric": cfg.get("is_metric", True),
                "unit": "meters" if cfg.get("is_metric", True) else "relative"
            }
        else:
            input_tensor_name = "images"
            output_tensor_name = "output0"
            classes_count = model_item.classes
            class_names = model_item.class_names

            # Detect whether model has End2End output [1, 300, 6]
            is_end2end = cfg.get("is_end2end", False)
            if not is_end2end and prepared_path and prepared_path.exists():
                name_low = prepared_path.name.lower()
                if "run6" in name_low or "nav-291" in name_low or "nav-408" in name_low or "nav-728" in name_low or "base_" in name_low or "official_yolo26" in name_low or "yolo26m" in name_low or "yolo26n" in name_low or "yolo26s" in name_low:
                    is_end2end = True
                elif "24k" in name_low and resolution >= 480:
                    is_end2end = True

            if is_end2end:
                out_shape = [1, 300, 6]
            else:
                anchors_count = (resolution // 8) ** 2 + (resolution // 16) ** 2 + (resolution // 32) ** 2
                out_shape = [1, 4 + classes_count, anchors_count]

            mean_vals = [0.0, 0.0, 0.0]
            std_vals = [255.0, 255.0, 255.0]
            stride_val = 32
            dist_heuristic = {
                "strategy": "bbox_area_ratio",
                "tiers": [
                    {"min_ratio": 0.35, "max_ratio": 1.00, "label": "<1 m", "meters": 0.8},
                    {"min_ratio": 0.20, "max_ratio": 0.35, "label": "1–2 m", "meters": 1.5},
                    {"min_ratio": 0.10, "max_ratio": 0.20, "label": "3–5 m", "meters": 4.0},
                    {"min_ratio": 0.05, "max_ratio": 0.10, "label": "6–10 m", "meters": 8.0},
                    {"min_ratio": 0.00, "max_ratio": 0.05, "label": ">10 m", "meters": 12.0}
                ]
            }

        return ModelMetadataResponse(
            id=model_id,
            name=model_item.name,
            category=model_item.category,
            task=model_item.task,
            cdn_url=cdn_url,
            cdn_urls=cdn_urls,
            run_id=model_item.run_id,
            variant=model_item.variant,
            resolution=resolution,
            format="onnx",
            file_size_mb=file_size_mb,
            sha256=sha256,
            classes=classes_count,
            class_names=class_names,
            input_shape=[1, 3, resolution, resolution],
            input_tensor_name=input_tensor_name,
            output_tensor_name=output_tensor_name,
            output_shape=out_shape,
            stride=stride_val,
            mean=mean_vals,
            std=std_vals,
            distance_heuristic=dist_heuristic,
            variants=model_item.variants
        )

registry_service = ModelRegistryService()
