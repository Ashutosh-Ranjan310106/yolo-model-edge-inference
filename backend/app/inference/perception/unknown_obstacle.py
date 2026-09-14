"""
Unknown Obstacle Detection Module.
Detects significant unclassified depth structures inside or near the walking corridor
where YOLO has not detected any object. Never invents an identity; names them "Unidentified obstacle".
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from app.inference.config import PerceptionConfig, pipeline_config
from app.inference.spatial.direction import DirectionClassifier
from app.inference.spatial.distance import DistanceEstimator
from app.inference.spatial.corridor import WalkingCorridor
from app.inference.spatial.spatial_fusion import SpatialObject

class UnknownObstacleDetector:
    """Detects non-YOLO spatial obstacles in or near the walking path."""
    def __init__(self, config: PerceptionConfig = None):
        self.config = config or pipeline_config.perception
        self.direction_classifier = DirectionClassifier()
        self.distance_estimator = DistanceEstimator()
        self.corridor = WalkingCorridor()

    def detect(
        self,
        depth_map: Optional[np.ndarray],
        lb_info: Optional[Dict[str, Any]],
        yolo_objects: List[SpatialObject],
        image_width: float,
        image_height: float
    ) -> List[SpatialObject]:
        """
        Scans walking corridor for depth clusters not explained by YOLO bounding boxes.
        Returns list of SpatialObject instances with class_name="Unidentified obstacle".
        """
        if depth_map is None or lb_info is None:
            return []

        scale = lb_info["scale"]
        pad_x = lb_info["pad_x"]
        pad_y = lb_info["pad_y"]
        new_w = lb_info["new_w"]
        new_h = lb_info["new_h"]
        cfg = self.config

        # 1. Create mask of existing YOLO bounding boxes on depth map
        known_mask = np.zeros(depth_map.shape, dtype=np.uint8)
        for obj in yolo_objects:
            bx1, by1, bx2, by2 = obj.bbox
            tx1 = int(round(bx1 * scale + pad_x))
            ty1 = int(round(by1 * scale + pad_y))
            tx2 = int(round(bx2 * scale + pad_x))
            ty2 = int(round(by2 * scale + pad_y))
            tx1 = max(0, min(depth_map.shape[1], tx1))
            tx2 = max(0, min(depth_map.shape[1], tx2))
            ty1 = max(0, min(depth_map.shape[0], ty1))
            ty2 = max(0, min(depth_map.shape[0], ty2))
            known_mask[ty1:ty2, tx1:tx2] = 255

        # 2. Extract walking corridor ROI in depth map
        # Central horizontal 60% and vertical 35% to 85%
        cy1 = int(pad_y + new_h * 0.35)
        cy2 = int(pad_y + new_h * 0.85)
        cx1 = int(pad_x + new_w * 0.20)
        cx2 = int(pad_x + new_w * 0.80)

        roi_depth = depth_map[cy1:cy2, cx1:cx2]
        roi_known = known_mask[cy1:cy2, cx1:cx2]

        # 3. Look for close depth structures (< unknown_max_depth_m) where YOLO detected nothing
        obstacle_mask = (
            (roi_depth >= cfg.unknown_min_depth_m) &
            (roi_depth <= cfg.unknown_max_depth_m) &
            (roi_known == 0)
        ).astype(np.uint8)

        # Morphological clean up to eliminate small sensor noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)

        # 4. Connected components to find significant spatial blobs
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_blob_area = (roi_depth.shape[0] * roi_depth.shape[1]) * cfg.unknown_min_blob_area_ratio

        unknown_objects: List[SpatialObject] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_blob_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Map back to letterbox coordinates
            abs_tx1 = cx1 + bx
            abs_ty1 = cy1 + by
            abs_tx2 = abs_tx1 + bw
            abs_ty2 = abs_ty1 + bh

            # Extract median depth inside blob
            blob_patch = roi_depth[by:by + bh, bx:bx + bw]
            blob_vals = blob_patch[blob_patch > 0]
            if len(blob_vals) < 10:
                continue

            med_dist = float(np.median(blob_vals))

            # Map to original camera coordinates
            orig_x1 = (abs_tx1 - pad_x) / scale
            orig_y1 = (abs_ty1 - pad_y) / scale
            orig_x2 = (abs_tx2 - pad_x) / scale
            orig_y2 = (abs_ty2 - pad_y) / scale

            orig_x1 = max(0.0, min(image_width, orig_x1))
            orig_y1 = max(0.0, min(image_height, orig_y1))
            orig_x2 = max(0.0, min(image_width, orig_x2))
            orig_y2 = max(0.0, min(image_height, orig_y2))

            blob_cx = (orig_x1 + orig_x2) / 2.0
            direction, h_pos = self.direction_classifier.classify(blob_cx, image_width)
            dist_m, dist_cat = self.distance_estimator.estimate_from_depth(med_dist, is_metric=True)
            overlap, relevance = self.corridor.compute_overlap(
                (orig_x1, orig_y1, orig_x2, orig_y2), image_width, image_height
            )

            if overlap >= cfg.unknown_corridor_relevance_min:
                unknown_objects.append(SpatialObject(
                    id=None,
                    type="unknown_obstacle",
                    class_name="Unidentified obstacle",
                    confidence=0.80,
                    bbox=[round(orig_x1, 1), round(orig_y1, 1), round(orig_x2, 1), round(orig_y2, 1)],
                    direction=direction,
                    horizontal_position=h_pos,
                    distance_m=dist_m,
                    distance_category=dist_cat,
                    corridor_overlap=overlap,
                    path_relevance=relevance,
                    depth_confidence=0.85
                ))

        return unknown_objects
