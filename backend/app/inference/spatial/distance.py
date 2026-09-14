"""
Distance Estimation Module.
Maps depth samples to calibrated metric distance and 5 standard distance categories:
VERY_NEAR, NEAR, MEDIUM, FAR, VERY_FAR.
"""

from typing import Tuple, Optional
from app.inference.config import DistanceConfig, pipeline_config

class DistanceEstimator:
    """Estimates distance in meters and standard navigation distance categories."""
    def __init__(self, config: DistanceConfig = None):
        self.config = config or pipeline_config.distance

    def categorize_metric(self, distance_m: float) -> str:
        """Categorizes metric distance into standard tiers."""
        cfg = self.config
        if distance_m < cfg.very_near_max_m:
            return "VERY_NEAR"
        elif distance_m < cfg.near_max_m:
            return "NEAR"
        elif distance_m < cfg.medium_max_m:
            return "MEDIUM"
        elif distance_m <= cfg.far_max_m:
            return "FAR"
        else:
            return "VERY_FAR"

    def estimate_from_depth(
        self,
        raw_depth_value: Optional[float],
        is_metric: bool = True
    ) -> Tuple[Optional[float], str]:
        """
        Estimates distance in meters and distance category.
        Returns:
            distance_m: float or None
            distance_category: "VERY_NEAR" | "NEAR" | "MEDIUM" | "FAR" | "VERY_FAR"
        """
        if raw_depth_value is None or raw_depth_value <= 0:
            return None, "FAR"

        if is_metric:
            distance_m = round(float(raw_depth_value), 2)
            category = self.categorize_metric(distance_m)
            return distance_m, category
        else:
            # Relative depth (0.0 to 1.0 or inverted)
            # Normalize to 0-10m scale for category assignment
            rel = max(0.01, min(10.0, float(raw_depth_value)))
            category = self.categorize_metric(rel)
            return round(rel, 2), category

    def estimate_from_bbox_ratio(self, area_ratio: float) -> Tuple[float, str]:
        """
        Fallback heuristic when depth is completely unavailable.
        Uses bounding box to frame area ratio.
        """
        cfg = self.config
        if area_ratio >= cfg.bbox_ratio_very_near:
            return 0.8, "VERY_NEAR"
        elif area_ratio >= cfg.bbox_ratio_near:
            return 1.5, "NEAR"
        elif area_ratio >= cfg.bbox_ratio_medium:
            return 3.0, "MEDIUM"
        elif area_ratio >= cfg.bbox_ratio_far:
            return 7.0, "FAR"
        else:
            return 12.0, "VERY_FAR"
