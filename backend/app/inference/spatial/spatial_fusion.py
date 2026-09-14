"""
Spatial Fusion Module.
Synthesizes 2D YOLO detections, Depth maps, 7-zone direction, and Walking corridor
into unified SpatialObject representations. Does NOT make final safety decisions.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from app.inference.spatial.direction import DirectionClassifier
from app.inference.spatial.distance import DistanceEstimator
from app.inference.spatial.corridor import WalkingCorridor

@dataclass
class SpatialObject:
    id: Optional[int]
    type: str                  # "object" | "wall" | "unknown_obstacle"
    class_name: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2]
    direction: str             # "far left" .. "far right"
    horizontal_position: float # [-1.0 .. 1.0]
    distance_m: Optional[float]
    distance_category: str     # "VERY_NEAR" .. "VERY_FAR"
    corridor_overlap: float    # [0.0 .. 1.0]
    path_relevance: str        # "outside path" | "partially relevant" | "likely in path"
    depth_confidence: float    # [0.0 .. 1.0]
    track_id: Optional[int] = None
    frames_seen: int = 1

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return d

class SpatialFusion:
    """Combines detection, depth, direction, and corridor overlap into SpatialObjects."""
    def __init__(self):
        self.direction_classifier = DirectionClassifier()
        self.distance_estimator = DistanceEstimator()
        self.corridor = WalkingCorridor()

    def fuse(
        self,
        detections: List[Dict[str, Any]],
        depth_estimator, # DepthEstimator instance
        depth_map: Optional[np.ndarray],
        lb_info: Optional[Dict[str, Any]],
        image_width: float,
        image_height: float
    ) -> List[SpatialObject]:
        """
        Processes detections and builds SpatialObject instances.
        """
        spatial_objects: List[SpatialObject] = []

        for det in detections:
            x1 = det["x1"]
            y1 = det["y1"]
            x2 = det["x2"]
            y2 = det["y2"]
            cx = det["center_x"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            box = (x1, y1, x2, y2)

            # 1. Direction & Horizontal Position
            direction, h_pos = self.direction_classifier.classify(cx, image_width)

            # 2. Distance from Depth map (with fallback)
            dist_m = None
            depth_conf = 0.50
            if depth_estimator is not None and depth_map is not None:
                dist_m, depth_conf = depth_estimator.get_roi_median_depth(
                    box, depth_map=depth_map, lb_info=lb_info
                )

            if dist_m is not None:
                dist_m, dist_cat = self.distance_estimator.estimate_from_depth(
                    dist_m, is_metric=depth_estimator.is_metric if depth_estimator else True
                )
            else:
                # Fallback to bounding-box area ratio heuristic
                box_area = max(0.0, (x2 - x1) * (y2 - y1))
                frame_area = image_width * image_height
                ratio = box_area / frame_area if frame_area > 0 else 0.0
                dist_m, dist_cat = self.distance_estimator.estimate_from_bbox_ratio(ratio)
                depth_conf = 0.35 # Lower confidence for bbox heuristic

            # 3. Corridor Overlap & Path Relevance
            overlap, relevance = self.corridor.compute_overlap(box, image_width, image_height)

            obj = SpatialObject(
                id=None,
                type="object",
                class_name=cls_name,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                direction=direction,
                horizontal_position=h_pos,
                distance_m=dist_m,
                distance_category=dist_cat,
                corridor_overlap=overlap,
                path_relevance=relevance,
                depth_confidence=depth_conf
            )
            spatial_objects.append(obj)

        return spatial_objects
