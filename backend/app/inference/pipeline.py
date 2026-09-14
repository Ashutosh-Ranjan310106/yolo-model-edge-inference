"""
NavigationPipeline: Master Perception & Guidance Pipeline for Edge Inference.
Coordinates frame-by-frame:
Camera Frame -> YOLO26 ONNX -> Depth ONNX -> Spatial Fusion -> Direction + Distance ->
Walking Corridor Analysis -> Ground / Wall / Unknown Analysis -> Object Tracking ->
Temporal Smoothing -> Risk Engine -> Priority Manager -> Guidance/Event Manager.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import cv2

from app.inference.config import PipelineConfig, pipeline_config
from app.inference.detector.yolo_detector import YOLODetector
from app.inference.depth.depth_estimator import DepthEstimator
from app.inference.spatial.spatial_fusion import SpatialFusion, SpatialObject
from app.inference.spatial.corridor import WalkingCorridor
from app.inference.perception.ground_analyzer import GroundAnalyzer
from app.inference.perception.wall_analyzer import WallAnalyzer
from app.inference.perception.unknown_obstacle import UnknownObstacleDetector
from app.inference.tracking.object_tracker import ObjectTracker
from app.inference.risk.risk_engine import RiskEngine
from app.inference.risk.priority_manager import PriorityManager
from app.inference.guidance.event_manager import EventManager

class NavigationPipeline:
    """Master frame-by-frame perception and deterministic guidance pipeline."""
    def __init__(
        self,
        config: PipelineConfig = None,
        yolo_model_id: str = "base_yolo26n",
        yolo_resolution: int = 480,
        depth_model_id: str = "yolo26n_depth",
        depth_resolution: int = 512,
        enable_depth: bool = True
    ):
        self.config = config or pipeline_config
        self.enable_depth = enable_depth

        # Core Modules
        self.detector = YOLODetector(model_id=yolo_model_id, resolution=yolo_resolution)
        self.depth_estimator = DepthEstimator(model_id=depth_model_id, resolution=depth_resolution) if enable_depth else None

        self.spatial_fusion = SpatialFusion()
        self.corridor = WalkingCorridor(self.config.corridor)
        self.ground_analyzer = GroundAnalyzer(self.config.perception)
        self.wall_analyzer = WallAnalyzer(self.config.perception)
        self.unknown_detector = UnknownObstacleDetector(self.config.perception)

        self.tracker = ObjectTracker(self.config.tracking)
        self.risk_engine = RiskEngine(self.config.risk)
        self.priority_manager = PriorityManager(self.config.risk)
        self.event_manager = EventManager(self.config.guidance)

        # Telemetry & Benchmark tracking
        self.frame_counter = 0
        self.frame_times = []
        self.fps = 0.0

    def configure(
        self,
        yolo_model_id: Optional[str] = None,
        yolo_resolution: Optional[int] = None,
        depth_model_id: Optional[str] = None,
        depth_resolution: Optional[int] = None,
        enable_depth: Optional[bool] = None
    ) -> bool:
        """Dynamically reconfigures model sessions on-the-fly without recreation."""
        if enable_depth is not None:
            self.enable_depth = enable_depth

        success = True
        if yolo_model_id or yolo_resolution:
            m_id = yolo_model_id or self.detector.model_id
            res = yolo_resolution or self.detector.resolution
            success = self.detector.load_model(m_id, res) and success

        if self.enable_depth:
            if self.depth_estimator is None:
                d_id = depth_model_id or "yolo26n_depth"
                d_res = depth_resolution or 512
                self.depth_estimator = DepthEstimator(model_id=d_id, resolution=d_res)
            elif depth_model_id or depth_resolution:
                d_id = depth_model_id or self.depth_estimator.model_id
                d_res = depth_resolution or self.depth_estimator.resolution
                success = self.depth_estimator.load_model(d_id, d_res) and success

        return success

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float = 0.25,
        frame_id: int = 0
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Executes complete perception and guidance pipeline for one frame.
        Returns:
            result_dict: Structured JSON representation ({ timestamp, fps, objects, events, latency })
            annotated_bgr: Visual debug frame marked with bounding boxes, risk, direction, distance, corridor
        """
        t_pipeline_start = time.perf_counter()
        self.frame_counter += 1
        orig_h, orig_w = frame_bgr.shape[:2]

        # 1. Update rolling FPS
        now_time = time.perf_counter()
        self.frame_times.append(now_time)
        if len(self.frame_times) > 15:
            self.frame_times.pop(0)
        if len(self.frame_times) >= 2:
            dur = self.frame_times[-1] - self.frame_times[0]
            self.fps = round((len(self.frame_times) - 1) / dur, 1) if dur > 0 else 0.0

        # 2. YOLO26 Detection (ONNX Runtime)
        detections, yolo_lat = self.detector.detect(frame_bgr, confidence_threshold=confidence_threshold)
        t_yolo_end = time.perf_counter()

        # 3. Depth Estimation (ONNX Runtime)
        depth_map = None
        lb_info = None
        depth_lat_ms = 0.0
        if self.enable_depth and self.depth_estimator is not None:
            depth_map, lb_info, depth_lat_ms = self.depth_estimator.estimate(frame_bgr)
        t_depth_end = time.perf_counter()

        # 4. Spatial Fusion (Bounding Boxes + Direction + Distance + Corridor overlap)
        t_fusion_start = time.perf_counter()
        spatial_objects = self.spatial_fusion.fuse(
            detections=detections,
            depth_estimator=self.depth_estimator,
            depth_map=depth_map,
            lb_info=lb_info,
            image_width=float(orig_w),
            image_height=float(orig_h)
        )

        # 5. Wall Spatial Filtering
        spatial_objects = self.wall_analyzer.filter_and_evaluate(spatial_objects)

        # 6. Unknown Obstacle Detection (Non-YOLO Depth Structures in/near Corridor)
        if self.enable_depth and depth_map is not None:
            unknown_obstacles = self.unknown_detector.detect(
                depth_map=depth_map,
                lb_info=lb_info,
                yolo_objects=spatial_objects,
                image_width=float(orig_w),
                image_height=float(orig_h)
            )
            spatial_objects.extend(unknown_obstacles)

        # 7. Ground / Surface Discontinuity Analysis (Drops, Steps, Stairs)
        ground_events = []
        if self.enable_depth and depth_map is not None:
            ground_events = self.ground_analyzer.analyze(
                depth_map=depth_map,
                lb_info=lb_info,
                image_width=float(orig_w),
                image_height=float(orig_h)
            )

        # 8. Object Tracking & Temporal Smoothing (IoU association, EMA smoothing)
        tracked_objects = self.tracker.update(spatial_objects)

        # 9. Risk Engine & Priority Evaluation
        evaluated_objects: List[Dict[str, Any]] = []
        for obj in tracked_objects:
            crit, risk, is_override = self.risk_engine.evaluate(obj)
            priority = self.priority_manager.classify_priority(risk, is_critical_override=is_override)

            obj_dict = {
                "id": obj.id,
                "class": obj.class_name,
                "confidence": obj.confidence,
                "bbox": obj.bbox,
                "direction": obj.direction,
                "horizontal_position": obj.horizontal_position,
                "distance_m": obj.distance_m,
                "distance_category": obj.distance_category,
                "corridor_overlap": obj.corridor_overlap,
                "criticality": crit,
                "risk": risk,
                "priority": priority,
                "type": obj.type
            }
            evaluated_objects.append(obj_dict)

        # Incorporate Ground Events into risk hierarchy
        for gev in ground_events:
            g_crit = 1.00 if "drop" in gev["class_name"] else 0.85
            g_risk = 0.90 if gev["distance_category"] in ("VERY_NEAR", "NEAR") else 0.65
            g_priority = "CRITICAL" if g_risk >= 0.80 else "HIGH"
            evaluated_objects.append({
                "id": None,
                "class": gev["class_name"],
                "confidence": gev["confidence"],
                "bbox": [0, int(orig_h * 0.7), orig_w, orig_h],
                "direction": gev["direction"],
                "horizontal_position": 0.0,
                "distance_m": gev.get("distance_m"),
                "distance_category": gev["distance_category"],
                "corridor_overlap": 0.90,
                "criticality": g_crit,
                "risk": g_risk,
                "priority": g_priority,
                "type": gev["type"]
            })

        # 10. Priority Ranking (Most urgent first)
        ranked_objects = self.priority_manager.rank_objects(evaluated_objects)

        # 11. Guidance & Announcement Event Manager (Cooldowns + state transitions)
        guidance_events = self.event_manager.process(ranked_objects)
        t_fusion_end = time.perf_counter()

        fusion_lat_ms = round((t_fusion_end - t_fusion_start) * 1000, 1)
        total_lat_ms = round((time.perf_counter() - t_pipeline_start) * 1000, 1)

        # 12. Construct Output Structured Payload
        result_payload = {
            "timestamp": int(time.time() * 1000),
            "frame_id": frame_id,
            "fps": self.fps,
            "objects": ranked_objects,
            "events": guidance_events,
            "latency": {
                "yolo_inference_ms": yolo_lat.get("inference_ms", 0.0),
                "depth_inference_ms": depth_lat_ms,
                "fusion_ms": fusion_lat_ms,
                "total_pipeline_ms": total_lat_ms
            }
        }

        # 13. Render Visual Debug Overlay Frame
        annotated_bgr = self.render_debug_overlay(
            frame_bgr=frame_bgr,
            objects=ranked_objects,
            events=guidance_events,
            latencies=result_payload["latency"],
            fps=self.fps
        )

        return result_payload, annotated_bgr

    def render_debug_overlay(
        self,
        frame_bgr: np.ndarray,
        objects: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        latencies: Dict[str, float],
        fps: float
    ) -> np.ndarray:
        """
        Draws visual debug annotations on the frame:
        - Walking corridor trapezoid
        - Bounding boxes colored by priority
        - Object labels (Class, Conf, Direction, Distance Category, Meters, Risk)
        - Guidance announcement banner
        - Real-time latency benchmark HUD
        """
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Walking Corridor Trapezoid (semi-transparent overlay)
        corridor_poly = self.corridor.get_corridor_polygon(w, h)
        corridor_pts = np.array([[int(p[0]), int(p[1])] for p in corridor_poly], np.int32)
        corridor_overlay = annotated.copy()
        cv2.fillPoly(corridor_overlay, [corridor_pts], (255, 230, 0)) # cyan-blue tint
        cv2.addWeighted(corridor_overlay, 0.12, annotated, 0.88, 0, annotated)
        cv2.polylines(annotated, [corridor_pts], isClosed=True, color=(255, 215, 0), thickness=1, lineType=cv2.LINE_AA)

        # 2. Draw Objects
        priority_colors = {
            "CRITICAL": (40, 40, 240),   # Bright Red
            "HIGH": (0, 140, 255),       # Orange/Amber
            "MEDIUM": (0, 230, 115),     # Green/Cyan
            "LOW": (200, 200, 200),      # Gray
            "IGNORE": (120, 120, 120)    # Dark Gray
        }

        for obj in objects:
            if obj.get("type") in ("possible_drop", "ground_rise"):
                continue

            box = obj.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in box]
            cls_name = obj.get("class", "Object")
            conf = obj.get("confidence", 0.0)
            direction = obj.get("direction", "center")
            dist_cat = obj.get("distance_category", "")
            dist_m = obj.get("distance_m")
            risk = obj.get("risk", 0.0)
            priority = obj.get("priority", "LOW")

            color = priority_colors.get(priority, (0, 230, 115))
            thickness = 3 if priority == "CRITICAL" else 2

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw multi-line label banner
            m_str = f"{dist_m:.1f}m" if dist_m is not None else ""
            label_line1 = f"{cls_name} {conf:.2f} | {direction.upper()}"
            label_line2 = f"{dist_cat} {m_str} | Risk: {risk:.2f}"

            (tw1, th1), _ = cv2.getTextSize(label_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            (tw2, th2), _ = cv2.getTextSize(label_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            banner_w = max(tw1, tw2) + 8
            banner_h = th1 + th2 + 10

            bg_y1 = max(0, y1 - banner_h)
            bg_y2 = y1
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + banner_w, bg_y2), color, -1)

            cv2.putText(annotated, label_line1, (x1 + 4, bg_y1 + th1 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated, label_line2, (x1 + 4, bg_y1 + th1 + th2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Top Telemetry Benchmark HUD
        yolo_ms = latencies.get("yolo_inference_ms", 0.0)
        depth_ms = latencies.get("depth_inference_ms", 0.0)
        fusion_ms = latencies.get("fusion_ms", 0.0)
        total_ms = latencies.get("total_pipeline_ms", 0.0)

        hud_text = f"FPS: {fps:.1f} | YOLO: {yolo_ms:.0f}ms | Depth: {depth_ms:.0f}ms | Fusion: {fusion_ms:.0f}ms | Total: {total_ms:.0f}ms"
        (ht_w, ht_h), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(annotated, (0, 0), (w, ht_h + 16), (15, 23, 42), -1)
        cv2.putText(annotated, hud_text, (10, ht_h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 1, cv2.LINE_AA)

        # 4. Guidance Banner (if active guidance event)
        if events:
            top_event = events[0]
            msg = top_event.get("message", "")
            prio = top_event.get("priority", "MEDIUM")
            banner_color = (40, 40, 240) if prio == "CRITICAL" else (0, 140, 255)
            cv2.rectangle(annotated, (0, h - 45), (w, h), banner_color, -1)
            cv2.putText(annotated, f"GUIDANCE: {msg}", (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

# Global master pipeline instance for backend
navigation_pipeline = NavigationPipeline()
