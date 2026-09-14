"""
Depth Estimator Module using ONNX Runtime.
Executes depth estimation independently from YOLO.
Handles letterboxing, inference, mapping to camera coordinates, and sampling.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import time
import numpy as np
import cv2
import onnxruntime as ort

from app.config.settings import settings
from app.services.registry import registry_service

class DepthEstimator:
    """
    ONNX Runtime Depth Estimator for edge navigation.
    Supports YOLO26-Depth (512, 320, 768) and Depth Anything V2.
    """
    def __init__(self, model_id: str = "yolo26n_depth", resolution: int = 512):
        self.model_id = model_id
        self.resolution = resolution
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "images"
        self.output_name: str = "depth"
        self.is_metric: bool = True
        self.mean: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std: np.ndarray = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        self.device: str = "CPU"
        self.latest_depth_map: Optional[np.ndarray] = None
        self.latest_letterbox_info: Optional[Dict[str, Any]] = None
        self.load_model(model_id, resolution)

    def resolve_onnx_path(self, model_id: str, resolution: int) -> Optional[Path]:
        """Finds depth ONNX model file."""
        # 1. Check resolution-specific depth models first (320, 512)
        if resolution in (320, 512):
            cand = settings.NAVIGATION_DIR / "Edge_Inference" / "models" / "depth" / f"depth_{resolution}.onnx"
            if cand.exists():
                return cand
            cand_mob = settings.NAVIGATION_DIR / "mobile_distributable_models" / f"depth_{resolution}.onnx"
            if cand_mob.exists():
                return cand_mob

        # 2. Check registry prepared path
        p = registry_service.get_prepared_model_path(model_id, resolution=resolution, format="onnx")
        if p and p.exists():
            return p

        # 3. Check Edge_Inference/models/depth/
        cand_direct = settings.NAVIGATION_DIR / "Edge_Inference" / "models" / "depth" / f"{model_id}.onnx"
        if cand_direct.exists():
            return cand_direct

        # 4. Check backend/weights/depth
        cand_w = settings.WEIGHTS_DIR / "depth" / f"{model_id}.onnx"
        if cand_w.exists():
            return cand_w

        return None

    def load_model(self, model_id: str, resolution: int = 512) -> bool:
        """Loads depth model ONNX session."""
        onnx_path = self.resolve_onnx_path(model_id, resolution)
        if not onnx_path or not onnx_path.exists():
            print(f"[DepthEstimator] Warning: Depth ONNX model '{model_id}' at {resolution}p not found.")
            return False

        # Providers
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "CUDA"
        else:
            self.device = "CPU"

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        t0 = time.perf_counter()
        try:
            self.session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
            self.model_id = model_id

            # Determine input name and adapt to actual fixed input shape if specified
            inputs = self.session.get_inputs()
            self.input_name = inputs[0].name
            in_shape = inputs[0].shape
            if len(in_shape) >= 4 and isinstance(in_shape[2], int) and in_shape[2] > 0:
                self.resolution = in_shape[2]
            else:
                self.resolution = resolution

            outputs = self.session.get_outputs()
            self.output_name = outputs[0].name

            meta = registry_service.get_model_metadata(model_id, resolution)
            if meta and meta.distance_heuristic:
                self.is_metric = meta.distance_heuristic.get("is_metric", True)
            else:
                self.is_metric = "yolo" in model_id.lower()

            if "yolo" in model_id.lower():
                self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                self.std = np.array([255.0, 255.0, 255.0], dtype=np.float32)
            else:
                self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
                self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[DepthEstimator] Loaded {onnx_path.name} ({resolution}x{resolution}, metric={self.is_metric}) on {self.device} in {elapsed:.1f} ms.")
            return True
        except Exception as e:
            print(f"[DepthEstimator] Error loading depth ONNX session: {e}")
            return False

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aspect-ratio preserving letterbox padding to resolution x resolution with neutral gray 114.
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        res = self.resolution

        scale = min(res / orig_w, res / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_x = (res - new_w) // 2
        pad_y = (res - new_h) // 2

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((res, res, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalize
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - self.mean) / self.std
        chw = np.transpose(norm, (2, 0, 1))
        nchw = np.expand_dims(chw, axis=0) # [1, 3, res, res]

        lb_info = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "new_w": new_w,
            "new_h": new_h,
            "res": res,
            "orig_w": orig_w,
            "orig_h": orig_h
        }

        return nchw, lb_info

    def estimate(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any], float]:
        """
        Runs depth inference.
        Returns:
            depth_map: 2D numpy array [res, res] containing depth values
            lb_info: letterbox metadata for coordinate conversion
            latency_ms: execution time in milliseconds
        """
        if self.session is None:
            return None, {}, 0.0

        t0 = time.perf_counter()
        tensor_input, lb_info = self.preprocess(frame_bgr)
        outputs = self.session.run([self.output_name], {self.input_name: tensor_input})
        latency_ms = (time.perf_counter() - t0) * 1000

        raw_depth = outputs[0]
        # Format might be [1, 1, H, W] or [1, H, W]
        if raw_depth.ndim == 4:
            depth_2d = raw_depth[0, 0]
        elif raw_depth.ndim == 3:
            depth_2d = raw_depth[0]
        else:
            depth_2d = raw_depth

        self.latest_depth_map = depth_2d
        self.latest_letterbox_info = lb_info

        return depth_2d, lb_info, round(latency_ms, 1)

    def get_roi_median_depth(
        self,
        box: Tuple[float, float, float, float],
        depth_map: Optional[np.ndarray] = None,
        lb_info: Optional[Dict[str, Any]] = None,
        crop_margin: float = 0.15
    ) -> Tuple[Optional[float], float]:
        """
        Samples central ~70% of bounding box, filters outliers, and computes median depth.
        Returns (median_depth, depth_confidence).
        """
        dmap = depth_map if depth_map is not None else self.latest_depth_map
        lb = lb_info if lb_info is not None else self.latest_letterbox_info

        if dmap is None or lb is None:
            return None, 0.0

        x1, y1, x2, y2 = box
        scale = lb["scale"]
        pad_x = lb["pad_x"]
        pad_y = lb["pad_y"]
        new_w = lb["new_w"]
        new_h = lb["new_h"]

        # Map box coordinates from original camera frame to letterboxed tensor pixels
        t_x1 = int(round(x1 * scale + pad_x))
        t_y1 = int(round(y1 * scale + pad_y))
        t_x2 = int(round(x2 * scale + pad_x))
        t_y2 = int(round(y2 * scale + pad_y))

        # Clamp to active letterbox area
        t_x1 = max(pad_x, min(pad_x + new_w - 1, t_x1))
        t_x2 = max(pad_x, min(pad_x + new_w - 1, t_x2))
        t_y1 = max(pad_y, min(pad_y + new_h - 1, t_y1))
        t_y2 = max(pad_y, min(pad_y + new_h - 1, t_y2))

        bw = t_x2 - t_x1
        bh = t_y2 - t_y1
        if bw < 3 or bh < 3:
            return None, 0.0

        # Sample inner core (ignoring outer crop_margin)
        margin_x = int(bw * crop_margin)
        margin_y = int(bh * crop_margin)
        core_x1 = t_x1 + margin_x
        core_x2 = max(core_x1 + 1, t_x2 - margin_x)
        core_y1 = t_y1 + margin_y
        core_y2 = max(core_y1 + 1, t_y2 - margin_y)

        core_patch = dmap[core_y1:core_y2, core_x1:core_x2]
        valid_vals = core_patch[np.isfinite(core_patch) & (core_patch > 0.0)]

        if len(valid_vals) < 5:
            return None, 0.0

        # Filter out extreme outliers (5th - 95th percentile)
        p5, p95 = np.percentile(valid_vals, [5, 95])
        inliers = valid_vals[(valid_vals >= p5) & (valid_vals <= p95)]
        if len(inliers) == 0:
            inliers = valid_vals

        median_val = float(np.median(inliers))
        std_val = float(np.std(inliers))

        # Depth confidence inversely proportional to relative variance
        confidence = 1.0 / (1.0 + (std_val / (median_val + 1e-4)))
        confidence = float(np.clip(confidence, 0.3, 0.99))

        return round(median_val, 2), round(confidence, 2)
