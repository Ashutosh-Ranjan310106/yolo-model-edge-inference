"""
Wall Analyzer Module.
Evaluates spatial wall detections by combining distance, direction, and corridor overlap.
Suppresses distant peripheral walls and prioritizes walls blocking the walking path.
"""

from typing import List
from app.inference.config import PerceptionConfig, pipeline_config
from app.inference.spatial.spatial_fusion import SpatialObject

class WallAnalyzer:
    """Filters and assesses spatial risk of walls in the environment."""
    def __init__(self, config: PerceptionConfig = None):
        self.config = config or pipeline_config.perception

    def filter_and_evaluate(self, spatial_objects: List[SpatialObject]) -> List[SpatialObject]:
        """
        Adjusts criticality and relevance of wall objects:
        - Suppresses announcement of distant/peripheral walls.
        - Boosts urgency of walls blocking the walking corridor.
        """
        cfg = self.config
        filtered_objects: List[SpatialObject] = []

        for obj in spatial_objects:
            if "wall" in obj.class_name.lower():
                # Check if wall is relevant to walking corridor
                dist_m = obj.distance_m or 5.0
                is_blocking = (
                    obj.corridor_overlap >= cfg.wall_min_corridor_overlap and
                    dist_m <= cfg.wall_max_announce_distance_m
                )

                if is_blocking:
                    obj.type = "wall"
                    filtered_objects.append(obj)
                else:
                    # Distant or peripheral wall: keep for debug tracking but mark as low relevance
                    obj.path_relevance = "outside path"
                    # Only retain if not excessively far
                    if dist_m <= 4.0:
                        filtered_objects.append(obj)
            else:
                filtered_objects.append(obj)

        return filtered_objects
