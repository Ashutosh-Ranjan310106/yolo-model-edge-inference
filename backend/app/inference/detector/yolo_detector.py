"""
YOLO Detector Module using ONNX Runtime.
Exclusively responsible for 2D object detection:
Returns class_id, class_name, confidence, x1, y1, x2, y2, center_x, center_y, width, height.
Does NOT compute distance or guidance.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import numpy as np
import cv2
import onnxruntime as ort

from app.config.settings import settings
from app.services.registry import registry_service

class YOLODetector:
    """
    ONNX Runtime YOLO Detector for real-time edge navigation.
    Maintains reusable session across frames without reloading.
    """
    def __init__(self, model_id: str = "base_yolo26n", resolution: int = 480):
        self.model_id = model_id
        self.resolution = resolution
        self.session: Optional[ort.InferenceSession] = None
        self.class_names: Dict[int, str] = {}
        self.input_name: str = "images"
        self.output_name: str = "output0"
        self.is_end2end: bool = False
        self.device: str = "CPU"
        self.load_model(model_id, resolution)

    def resolve_onnx_path(self, model_id: str, resolution: int) -> Optional[Path]:
        """Finds ONNX model path from registry or known paths."""
        # 1. Check registry prepared path
        p = registry_service.get_prepared_model_path(model_id, resolution=resolution, format="onnx")
        if p and p.exists():
            return p

        # 2. Check Edge_Inference/models/
        cand1 = settings.NAVIGATION_DIR / "Edge_Inference" / "models" / f"{model_id}_{resolution}.onnx"
        if cand1.exists():
            return cand1
        cand2 = settings.NAVIGATION_DIR / "Edge_Inference" / "models" / f"{model_id}.onnx"
        if cand2.exists():
            return cand2

        # 3. Check mobile_distributable_models
        cand3 = settings.NAVIGATION_DIR / "mobile_distributable_models" / f"{model_id}_{resolution}.onnx"
        if cand3.exists():
            return cand3
        cand4 = settings.NAVIGATION_DIR / "mobile_distributable_models" / f"{model_id}.onnx"
        if cand4.exists():
            return cand4

        # 4. Check backend/weights/
        cand5 = settings.WEIGHTS_DIR / f"{model_id}.onnx"
        if cand5.exists():
            return cand5

        return None

    def load_model(self, model_id: str, resolution: int = 480) -> bool:
        """Loads or switches active ONNX model session."""
        onnx_path = self.resolve_onnx_path(model_id, resolution)
        if not onnx_path or not onnx_path.exists():
            print(f"[YOLODetector] Warning: ONNX model '{model_id}' at {resolution}p not found on disk.")
            return False

        # Configure Execution Providers
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
            self.resolution = resolution

            # Introspect inputs & outputs
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            self.input_name = inputs[0].name
            self.output_name = outputs[0].name

            # Check if output is End2End [1, 300, 6] or Anchor format [1, 4+C, N]
            out_shape = outputs[0].shape
            self.is_end2end = (len(out_shape) == 3 and out_shape[1] == 300 and out_shape[2] == 6) or \
                              (len(out_shape) == 3 and out_shape[2] == 6)

            # Retrieve class names from registry
            meta = registry_service.get_model_metadata(model_id, resolution)
            if meta and meta.class_names:
                self.class_names = {i: name for i, name in enumerate(meta.class_names)}
            else:
                self.class_names = {i: f"class_{i}" for i in range(80)}

            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[YOLODetector] Loaded {onnx_path.name} ({resolution}x{resolution}) on {self.device} in {elapsed:.1f} ms. End2End={self.is_end2end}")
            return True
        except Exception as e:
            print(f"[YOLODetector] Error loading ONNX model {onnx_path}: {e}")
            return False

    def preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Letterbox pad frame to resolution x resolution with neutral gray (114, 114, 114).
        Returns normalized float32 tensor (NCHW) and letterbox metadata (scale, pad_x, pad_y).
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        res = self.resolution

        scale = min(res / orig_w, res / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_x = (res - new_w) // 2
        pad_y = (res - new_h) // 2

        # Resize image
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad canvas with 114
        canvas = np.full((res, res, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # HWC -> CHW, BGR -> RGB, normalize 0.0 - 1.0
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        nchw = np.expand_dims(chw, axis=0) # [1, 3, res, res]

        return nchw, scale, pad_x, pad_y

    def detect(self, frame_bgr: np.ndarray, confidence_threshold: float = 0.25) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Executes YOLO inference.
        Returns:
            detections: List of dicts containing:
                class_id, class_name, confidence, x1, y1, x2, y2, center_x, center_y, width, height
            latencies: Dictionary of timing metrics (preprocess_ms, inference_ms, postprocess_ms)
        """
        if self.session is None:
            return [], {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "total_ms": 0.0}

        orig_h, orig_w = frame_bgr.shape[:2]

        t0 = time.perf_counter()
        tensor_input, scale, pad_x, pad_y = self.preprocess(frame_bgr)
        t_pre = time.perf_counter()

        outputs = self.session.run([self.output_name], {self.input_name: tensor_input})
        t_inf = time.perf_counter()

        raw_output = outputs[0]
        detections: List[Dict[str, Any]] = []

        if self.is_end2end:
            # End2End format: [1, 300, 6] -> [x1, y1, x2, y2, score, class_id]
            preds = raw_output[0]
            for pred in preds:
                conf = float(pred[4])
                if conf < confidence_threshold:
                    continue
                cls_id = int(pred[5])
                # Unletterbox coordinates to original frame pixels
                x1 = (pred[0] - pad_x) / scale
                y1 = (pred[1] - pad_y) / scale
                x2 = (pred[2] - pad_x) / scale
                y2 = (pred[3] - pad_y) / scale

                # Clamp to frame bounds
                x1 = max(0.0, min(float(orig_w), float(x1)))
                y1 = max(0.0, min(float(orig_h), float(y1)))
                x2 = max(0.0, min(float(orig_w), float(x2)))
                y2 = max(0.0, min(float(orig_h), float(y2)))

                box_w = max(0.0, x2 - x1)
                box_h = max(0.0, y2 - y1)
                if box_w <= 1 or box_h <= 1:
                    continue

                cx = x1 + box_w / 2.0
                cy = y1 + box_h / 2.0
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "center_x": round(cx, 1),
                    "center_y": round(cy, 1),
                    "width": round(box_w, 1),
                    "height": round(box_h, 1)
                })
        else:
            # Standard anchor format [1, 4+C, N]
            preds = raw_output[0] # [4+C, N]
            num_classes = preds.shape[0] - 4
            anchors = preds.shape[1]

            boxes_xyxy = []
            scores = []
            class_ids = []

            for a in range(anchors):
                class_scores = preds[4:, a]
                top_class_id = int(np.argmax(class_scores))
                top_score = float(class_scores[top_class_id])

                if top_score >= confidence_threshold:
                    cx_p, cy_p, w_p, h_p = preds[0, a], preds[1, a], preds[2, a], preds[3, a]
                    x1 = (cx_p - w_p / 2.0 - pad_x) / scale
                    y1 = (cy_p - h_p / 2.0 - pad_y) / scale
                    x2 = (cx_p + w_p / 2.0 - pad_x) / scale
                    y2 = (cy_p + h_p / 2.0 - pad_y) / scale

                    boxes_xyxy.append([x1, y1, x2, y2])
                    scores.append(top_score)
                    class_ids.append(top_class_id)

            if len(boxes_xyxy) > 0:
                # Apply NMS
                boxes_cv = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_xyxy]
                indices = cv2.dnn.NMSBoxes(boxes_cv, scores, confidence_threshold, 0.45)
                if len(indices) > 0:
                    for idx in indices.flatten():
                        bx = boxes_xyxy[idx]
                        x1 = max(0.0, min(float(orig_w), float(bx[0])))
                        y1 = max(0.0, min(float(orig_h), float(bx[1])))
                        x2 = max(0.0, min(float(orig_w), float(bx[2])))
                        y2 = max(0.0, min(float(orig_h), float(bx[3])))

                        box_w = max(0.0, x2 - x1)
                        box_h = max(0.0, y2 - y1)
                        if box_w <= 1 or box_h <= 1:
                            continue

                        cx = x1 + box_w / 2.0
                        cy = y1 + box_h / 2.0
                        cls_id = class_ids[idx]
                        cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(scores[idx], 3),
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                            "center_x": round(cx, 1),
                            "center_y": round(cy, 1),
                            "width": round(box_w, 1),
                            "height": round(box_h, 1)
                        })

        t_post = time.perf_counter()

        latencies = {
            "preprocess_ms": round((t_pre - t0) * 1000, 1),
            "inference_ms": round((t_inf - t_pre) * 1000, 1),
            "postprocess_ms": round((t_post - t_inf) * 1000, 1),
            "total_ms": round((t_post - t0) * 1000, 1)
        }

        return detections, latencies
