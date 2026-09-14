import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from app.config.settings import settings

class TestSessionRecorder:
    """
    Manages real-world test recordings under backend/runs/<timestamp>/
    Saves paired original (input/) and annotated (annotated/) frames by matching frame ID.
    Maintains metadata.json for the session.
    """
    def __init__(self, base_runs_dir: Optional[Path] = None):
        self.base_runs_dir = base_runs_dir or (settings.BASE_DIR / "runs")
        self.base_runs_dir.mkdir(parents=True, exist_ok=True)

        self.session_id: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.input_dir: Optional[Path] = None
        self.annotated_dir: Optional[Path] = None

        self.enabled: bool = False
        self.save_frequency: int = 5
        self.only_with_detections: bool = False
        self.frame_counter: int = 0
        self.saved_frames_count: int = 0

        self.metadata: Dict[str, Any] = {}
        self.saved_records: List[Dict[str, Any]] = []

    def start_session(
        self,
        model_name: str,
        resolution: int,
        confidence: float,
        enabled: bool = True,
        frequency: int = 5,
        only_detections: bool = False,
        device: str = "PC"
    ) -> Optional[str]:
        """Initializes a new recording session directory and metadata structure."""
        self.enabled = enabled
        self.save_frequency = max(1, frequency)
        self.only_with_detections = only_detections
        self.frame_counter = 0
        self.saved_frames_count = 0
        self.saved_records = []

        if not self.enabled:
            self.session_id = None
            self.session_dir = None
            return None

        now = datetime.now()
        self.session_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.base_runs_dir / self.session_id
        self.input_dir = self.session_dir / "input"
        self.annotated_dir = self.session_dir / "annotated"

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            "session_id": self.session_id,
            "test_start_time": now.isoformat() + "Z",
            "test_end_time": None,
            "inference_device": device,
            "model": model_name,
            "model_version": "1.0",
            "resolution": resolution,
            "confidence_threshold": confidence,
            "runtime": "PyTorch / Ultralytics",
            "runtime_backend": "CUDA" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "CPU",
            "frame_save_frequency": self.save_frequency,
            "only_with_detections": self.only_with_detections,
            "total_frames_processed": 0,
            "total_frames_saved": 0,
            "frames": []
        }

        self._flush_metadata()
        print(f"[Session Recorder] Started test session: {self.session_id}")
        return self.session_id

    def should_save_frame(self, detections_count: int) -> bool:
        """Determines if the current frame satisfies saving frequency and filters."""
        if not self.enabled or self.session_dir is None:
            return False

        self.frame_counter += 1
        if self.only_with_detections and detections_count == 0:
            return False

        return (self.frame_counter % self.save_frequency) == 0

    def draw_annotations(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float,
        latency_ms: float
    ) -> np.ndarray:
        """Draws bounding boxes, classes, confidence, and distances on a copy of the frame."""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw detections
        for det in detections:
            box = det.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in box]
            cls_name = det.get("class_name", "Object")
            conf = det.get("confidence", 0.0)
            dist = det.get("distance", "")

            # Color by class or distance (emerald for safe, amber for close, red for very close)
            color = (0, 230, 115) # default green
            if "<1" in dist:
                color = (40, 40, 240) # red
            elif "1-2" in dist:
                color = (0, 165, 255) # orange/amber

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{cls_name} {conf:.2f} | {dist}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 6)), (x1 + text_w + 4, max(text_h + 6, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 2, max(text_h + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Header HUD banner
        hud_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Objects: {len(detections)}"
        cv2.putText(annotated, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def record_frame(
        self,
        frame_id: int,
        timestamp_client: int,
        original_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float,
        latency_dict: Dict[str, float]
    ) -> bool:
        """Saves original and annotated pair with identical frame_id and appends metadata."""
        if not self.should_save_frame(len(detections)):
            return False

        filename = f"frame_{frame_id:06d}.jpg"
        input_path = self.input_dir / filename
        annotated_path = self.annotated_dir / filename

        # 1. Save original input image
        cv2.imwrite(str(input_path), original_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 2. Draw annotations and save
        total_lat = latency_dict.get("total_ms", 0.0)
        annotated_bgr = self.draw_annotations(original_bgr, detections, fps, total_lat)
        cv2.imwrite(str(annotated_path), annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 3. Log record
        record = {
            "frame_id": frame_id,
            "timestamp": timestamp_client,
            "fps": round(fps, 1),
            "inference_ms": latency_dict.get("inference_ms", 0.0),
            "total_latency_ms": total_lat,
            "detection_count": len(detections),
            "detections": detections,
            "input": f"input/{filename}",
            "annotated": f"annotated/{filename}"
        }
        self.saved_records.append(record)
        self.saved_frames_count += 1

        # Periodically flush metadata
        if (self.saved_frames_count % 10) == 0:
            self._flush_metadata()

        return True

    def _flush_metadata(self):
        """Writes current metadata state to disk."""
        if not self.session_dir:
            return
        meta_file = self.session_dir / "metadata.json"
        self.metadata["total_frames_processed"] = self.frame_counter
        self.metadata["total_frames_saved"] = self.saved_frames_count
        self.metadata["frames"] = self.saved_records
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def end_session(self) -> Optional[Dict[str, Any]]:
        """Finalizes the active session."""
        if not self.enabled or not self.session_dir:
            return None

        self.metadata["test_end_time"] = datetime.now().isoformat() + "Z"
        self._flush_metadata()
        print(f"[Session Recorder] Finished session '{self.session_id}'. Total saved: {self.saved_frames_count} frames.")
        summary = {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "total_saved": self.saved_frames_count
        }
        self.enabled = False
        return summary

session_recorder = TestSessionRecorder()
