import time
import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config.settings import settings
from app.services.registry import registry_service

class PCYOLORunner:
    """
    High-performance PC-side YOLO runner for real-time WebSocket inference.
    Loads PyTorch (.pt) weights once and reuses them across incoming video frames.
    """
    def __init__(self):
        self.current_model: Optional[YOLO] = None
        self.current_model_id: Optional[str] = None
        self.current_model_path: Optional[Path] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names: Dict[int, str] = {}
        print(f"[PC YOLO] Initialized runner with execution device: {self.device.upper()}")

    def resolve_pt_path(self, model_id: str) -> Optional[Path]:
        """Locates the source .pt weights for a model identifier."""
        # 1. Check registry / scan models
        item = registry_service.find_model_item(model_id)
        if item and item.source_path:
            p = Path(item.source_path)
            if p.exists():
                return p

        # 2. Check weights/original/
        orig = settings.WEIGHTS_DIR / "original" / f"{model_id}.pt"
        if orig.exists():
            return orig
        orig_direct = settings.WEIGHTS_DIR / "original" / model_id
        if orig_direct.exists():
            return orig_direct

        # 3. Check weights/
        w_cand = settings.WEIGHTS_DIR / f"{model_id}.pt"
        if w_cand.exists():
            return w_cand

        # 4. Check runs/train/<model_id>/weights/best.pt
        clean_id = model_id.replace("_best", "").replace("_last", "")
        run_best = settings.TRAINING_RUNS_DIR / clean_id / "weights" / "best.pt"
        if run_best.exists():
            return run_best

        # 5. Search recursively in runs/train/
        for p in settings.TRAINING_RUNS_DIR.glob(f"**/{model_id}.pt"):
            if p.is_file():
                return p

        # 6. Check navigation root
        nav_cand = settings.NAVIGATION_DIR / f"{model_id}.pt"
        if nav_cand.exists():
            return nav_cand

        return None

    def load_model(self, model_id: str) -> bool:
        """Loads a model into memory if not already active."""
        if self.current_model is not None and self.current_model_id == model_id:
            return True

        pt_path = self.resolve_pt_path(model_id)
        if not pt_path or not pt_path.exists():
            print(f"[PC YOLO] Error: Could not locate .pt weights for '{model_id}'")
            return False

        print(f"[PC YOLO] Loading model '{model_id}' from {pt_path.name} on {self.device.upper()}...")
        t0 = time.perf_counter()
        try:
            self.current_model = YOLO(str(pt_path))
            self.current_model.to(self.device)
            self.current_model_id = model_id
            self.current_model_path = pt_path
            self.class_names = self.current_model.names or {}
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[PC YOLO] Model '{model_id}' loaded successfully in {elapsed:.1f} ms with {len(self.class_names)} classes.")
            return True
        except Exception as e:
            print(f"[PC YOLO] Failed to load model {model_id}: {e}")
            return False

    @staticmethod
    def calculate_distance(box_area_ratio: float) -> Tuple[str, float]:
        """
        Estimate distance based on the bounding box area to frame area ratio.
        >35%   -> <1 m
        20-35% -> 1-2 m
        10-20% -> 3-5 m
        5-10%  -> 6-10 m
        <5%    -> >10 m
        """
        if box_area_ratio >= 0.35:
            return "<1 m", 0.8
        elif box_area_ratio >= 0.20:
            return "1-2 m", 1.5
        elif box_area_ratio >= 0.10:
            return "3-5 m", 2.8
        elif box_area_ratio >= 0.05:
            return "6-10 m", 7.5
        else:
            return ">10 m", 12.0

    def predict_frame(
        self,
        image_bytes: bytes,
        resolution: int = 480,
        confidence_threshold: float = 0.25
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
        """
        Executes YOLO inference on a JPEG frame buffer.
        Returns: (frame_bgr, detections_list, latencies_dict)
        """
        if self.current_model is None:
            raise RuntimeError("No model is currently loaded in PCYOLORunner.")

        t_start = time.perf_counter()

        # 1. Decode JPEG
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Failed to decode image bytes into valid frame.")

        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)

        t_pre_end = time.perf_counter()

        # 2. Run Ultralytics YOLO inference
        results = self.current_model.predict(
            source=frame_bgr,
            imgsz=resolution,
            conf=confidence_threshold,
            device=self.device,
            verbose=False
        )

        t_inf_end = time.perf_counter()

        # 3. Parse detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"Class_{cls_id}")

                x1, y1, x2, y2 = xyxy
                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                box_area = box_w * box_h
                area_ratio = box_area / frame_area if frame_area > 0 else 0.0

                dist_label, approx_m = self.calculate_distance(area_ratio)

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "distance": dist_label,
                    "approx_meters": approx_m,
                    "area_ratio": round(area_ratio, 4)
                })

        t_post_end = time.perf_counter()

        latencies = {
            "decode_ms": round((t_pre_end - t_start) * 1000, 1),
            "inference_ms": round((t_inf_end - t_pre_end) * 1000, 1),
            "postprocess_ms": round((t_post_end - t_inf_end) * 1000, 1),
            "total_ms": round((t_post_end - t_start) * 1000, 1)
        }

        return frame_bgr, detections, latencies

pc_runner = PCYOLORunner()
