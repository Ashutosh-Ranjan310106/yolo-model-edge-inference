"""
Direction Classification Module.
Calculates continuous normalized horizontal position in [-1.0, 1.0]
and assigns one of seven configurable directional zones.
"""

from typing import Tuple
from app.inference.config import DirectionConfig, pipeline_config

class DirectionClassifier:
    """Classifies horizontal orientation relative to camera center."""
    def __init__(self, config: DirectionConfig = None):
        self.config = config or pipeline_config.direction

    def classify(self, center_x: float, image_width: float) -> Tuple[str, float]:
        """
        Calculates normalized horizontal position and 7-zone direction.
        Returns:
            direction: str ("far left", "left", "slightly left", "center",
                            "slightly right", "right", "far right")
            horizontal_position: float in [-1.0, 1.0] (rounded to 2 decimal places)
        """
        if image_width <= 0:
            return "center", 0.0

        half_w = image_width / 2.0
        relative_x = (center_x - half_w) / half_w
        relative_x = max(-1.0, min(1.0, relative_x))
        h_pos = round(float(relative_x), 2)

        cfg = self.config
        if relative_x < cfg.far_left_max:
            direction = "far left"
        elif relative_x < cfg.left_max:
            direction = "left"
        elif relative_x < cfg.slightly_left_max:
            direction = "slightly left"
        elif relative_x <= cfg.center_max:
            direction = "center"
        elif relative_x <= cfg.slightly_right_max:
            direction = "slightly right"
        elif relative_x <= cfg.right_max:
            direction = "right"
        else:
            direction = "far right"

        return direction, h_pos
