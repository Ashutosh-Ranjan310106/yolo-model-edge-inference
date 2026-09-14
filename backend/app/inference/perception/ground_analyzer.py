"""
Ground Analyzer Module.
Analyzes the lower portion of the depth map independently from YOLO to detect
drops, rises, stairs, and uneven ground surfaces.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from app.inference.config import PerceptionConfig, pipeline_config
from app.inference.spatial.distance import DistanceEstimator

class GroundAnalyzer:
    """Detects surface anomalies (drop-offs, steps, uneven ground) from depth map."""
    def __init__(self, config: PerceptionConfig = None):
        self.config = config or pipeline_config.perception
        self.distance_estimator = DistanceEstimator()

    def analyze(
        self,
        depth_map: Optional[np.ndarray],
        lb_info: Optional[Dict[str, Any]],
        image_width: float,
        image_height: float
    ) -> List[Dict[str, Any]]:
        """
        Analyzes the ground region for depth discontinuities.
        Returns list of GroundEvent dicts.
        """
        if depth_map is None or lb_info is None:
            return []

        pad_x = lb_info["pad_x"]
        pad_y = lb_info["pad_y"]
        new_w = lb_info["new_w"]
        new_h = lb_info["new_h"]

        cfg = self.config
        y_start = int(pad_y + new_h * cfg.ground_band_y_start)
        y_end = int(pad_y + new_h * cfg.ground_band_y_end)
        x_start = int(pad_x + new_w * 0.25)
        x_end = int(pad_x + new_w * 0.75)

        ground_patch = depth_map[y_start:y_end, x_start:x_end]
        if ground_patch.size < 50:
            return []

        # Divide into 3 columns: left, center, right of walking path
        col_w = ground_patch.shape[1] // 3
        events: List[Dict[str, Any]] = []

        sectors = [
            ("slightly left", ground_patch[:, :col_w]),
            ("center", ground_patch[:, col_w:2 * col_w]),
            ("slightly right", ground_patch[:, 2 * col_w:])
        ]

        for sector_name, col_data in sectors:
            # Analyze vertical depth profile from near (bottom) to far (top)
            # In image: row 0 is far, bottom row is near
            valid_col = col_data[np.isfinite(col_data) & (col_data > 0)]
            if len(valid_col) < 20:
                continue

            # Check median depth in near half vs far half
            h_half = col_data.shape[0] // 2
            near_patch = col_data[h_half:, :]
            far_patch = col_data[:h_half, :]

            v_near = near_patch[np.isfinite(near_patch) & (near_patch > 0)]
            v_far = far_patch[np.isfinite(far_patch) & (far_patch > 0)]

            if len(v_near) < 10 or len(v_far) < 10:
                continue

            near_d = float(np.median(v_near))
            far_d = float(np.median(v_far))

            # Expected: near_d < far_d (ground recedes into distance)
            # 1. Sudden Drop / Hole: Depth suddenly becomes much deeper than expected
            if near_d > 0 and (far_d / (near_d + 1e-4)) > cfg.drop_depth_jump_ratio and near_d < 3.0:
                dist_cat = self.distance_estimator.categorize_metric(near_d)
                events.append({
                    "type": "possible_drop",
                    "class_name": "possible drop",
                    "direction": sector_name,
                    "distance_m": round(near_d, 2),
                    "distance_category": dist_cat,
                    "confidence": 0.85
                })
            # 2. Sudden Rise / Step Up: Near ground is deeper than an obstacle right ahead
            elif near_d > 0 and (near_d / (far_d + 1e-4)) > (1.0 / cfg.rise_depth_jump_ratio) and far_d < 2.5:
                dist_cat = self.distance_estimator.categorize_metric(far_d)
                events.append({
                    "type": "ground_rise",
                    "class_name": "ground rise",
                    "direction": sector_name,
                    "distance_m": round(far_d, 2),
                    "distance_category": dist_cat,
                    "confidence": 0.80
                })

        return events
