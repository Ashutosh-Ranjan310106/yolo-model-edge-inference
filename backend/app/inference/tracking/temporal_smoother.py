"""
Temporal Smoother Module.
Applies Exponential Moving Average (EMA) smoothing to continuous horizontal position
and metric distance to eliminate frame-to-frame flicker and noise.
"""

from typing import Tuple, Optional
from app.inference.config import TrackingConfig, pipeline_config
from app.inference.spatial.direction import DirectionClassifier
from app.inference.spatial.distance import DistanceEstimator

class TemporalSmoother:
    """Smooths object positions and distances over time using configurable EMA."""
    def __init__(self, config: TrackingConfig = None):
        self.config = config or pipeline_config.tracking
        self.direction_classifier = DirectionClassifier()
        self.distance_estimator = DistanceEstimator()

    def smooth_position(
        self,
        prev_h_pos: float,
        curr_h_pos: float,
        alpha: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Applies EMA smoothing to horizontal position in [-1.0, 1.0].
        Returns:
            smoothed_h_pos: float in [-1.0, 1.0]
            direction: updated 7-zone direction based on smoothed position
        """
        a = alpha if alpha is not None else self.config.alpha_x
        # smoothed = (1 - a) * prev + a * curr
        smoothed = (1.0 - a) * prev_h_pos + a * curr_h_pos
        smoothed = max(-1.0, min(1.0, smoothed))
        smoothed = round(float(smoothed), 2)

        # Re-derive 7-zone direction from smoothed continuous coordinate
        cfg = pipeline_config.direction
        if smoothed < cfg.far_left_max:
            direction = "far left"
        elif smoothed < cfg.left_max:
            direction = "left"
        elif smoothed < cfg.slightly_left_max:
            direction = "slightly left"
        elif smoothed <= cfg.center_max:
            direction = "center"
        elif smoothed <= cfg.slightly_right_max:
            direction = "slightly right"
        elif smoothed <= cfg.right_max:
            direction = "right"
        else:
            direction = "far right"

        return smoothed, direction

    def smooth_distance(
        self,
        prev_dist: Optional[float],
        curr_dist: Optional[float],
        is_metric: bool = True,
        alpha: Optional[float] = None
    ) -> Tuple[Optional[float], str]:
        """
        Applies EMA smoothing to distance in meters.
        Returns:
            smoothed_dist: float in meters
            distance_category: updated category based on smoothed distance
        """
        if curr_dist is None:
            if prev_dist is not None:
                return prev_dist, self.distance_estimator.categorize_metric(prev_dist)
            return None, "FAR"

        if prev_dist is None:
            return round(curr_dist, 2), self.distance_estimator.categorize_metric(curr_dist)

        a = alpha if alpha is not None else self.config.alpha_distance
        smoothed = (1.0 - a) * prev_dist + a * curr_dist
        smoothed = max(0.2, round(float(smoothed), 2))
        category = self.distance_estimator.categorize_metric(smoothed)

        return smoothed, category
