"""
Walking Corridor Module.
Models a trapezoidal walking corridor extending from bottom-center of image toward horizon.
Calculates polygon intersection overlap and path relevance.
"""

from typing import List, Tuple
from app.inference.config import CorridorConfig, pipeline_config
from app.inference.spatial.geometry import box_to_polygon, polygon_intersection, polygon_area

class WalkingCorridor:
    """Trapezoidal walking corridor for obstacle path relevance evaluation."""
    def __init__(self, config: CorridorConfig = None):
        self.config = config or pipeline_config.corridor

    def get_corridor_polygon(self, image_width: float, image_height: float) -> List[Tuple[float, float]]:
        """
        Returns the 4 vertices of the trapezoidal corridor in image pixels.
        Top: centered at horizon_y with top_width_ratio * image_width.
        Bottom: centered at bottom_y with bottom_width_ratio * image_width.
        """
        cfg = self.config
        cx = image_width / 2.0

        top_y = cfg.horizon_y * image_height
        bot_y = cfg.bottom_y * image_height

        top_half_w = (cfg.top_width_ratio * image_width) / 2.0
        bot_half_w = (cfg.bottom_width_ratio * image_width) / 2.0

        p1 = (cx - top_half_w, top_y) # Top Left
        p2 = (cx + top_half_w, top_y) # Top Right
        p3 = (cx + bot_half_w, bot_y) # Bottom Right
        p4 = (cx - bot_half_w, bot_y) # Bottom Left

        return [p1, p2, p3, p4]

    def compute_overlap(
        self,
        box: Tuple[float, float, float, float],
        image_width: float,
        image_height: float
    ) -> Tuple[float, str]:
        """
        Calculates corridor overlap ratio: intersection(box, corridor) / box_area.
        Returns:
            corridor_overlap: float in [0.0, 1.0]
            relevance_class: "outside path" | "partially relevant" | "likely in path"
        """
        x1, y1, x2, y2 = box
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        box_area = box_w * box_h

        if box_area <= 0:
            return 0.0, "outside path"

        box_poly = box_to_polygon(x1, y1, x2, y2)
        corridor_poly = self.get_corridor_polygon(image_width, image_height)

        inter_area = polygon_intersection(box_poly, corridor_poly)
        overlap_ratio = float(min(1.0, max(0.0, inter_area / box_area)))
        overlap_ratio = round(overlap_ratio, 2)

        cfg = self.config
        if overlap_ratio < cfg.outside_threshold:
            relevance = "outside path"
        elif overlap_ratio < cfg.partial_threshold:
            relevance = "partially relevant"
        else:
            relevance = "likely in path"

        return overlap_ratio, relevance
