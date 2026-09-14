"""
Risk Engine Module.
Calculates compound risk score from criticality, distance, corridor relevance,
confidence, and persistence. Enforces Critical Overrides for imminent life-safety hazards.
"""

from typing import Tuple, Dict, Any
from app.inference.config import RiskConfig, pipeline_config
from app.inference.spatial.spatial_fusion import SpatialObject
from app.inference.risk.criticality import CriticalityTable

class RiskEngine:
    """Calculates risk score [0.0 - 1.0] and handles critical safety overrides."""
    def __init__(self, config: RiskConfig = None):
        self.config = config or pipeline_config.risk
        self.criticality_table = CriticalityTable(self.config)

    def evaluate(self, obj: SpatialObject) -> Tuple[float, float, bool]:
        """
        Calculates criticality, compound risk, and critical override flag.
        Returns:
            criticality: float in [0.0, 1.0]
            risk: float in [0.0, 1.0]
            is_critical_override: bool
        """
        cfg = self.config
        crit = self.criticality_table.get(obj.class_name)

        # 1. Distance factor
        dist_factor = cfg.distance_factors.get(obj.distance_category, 0.20)

        # 2. Path relevance factor
        path_factor = cfg.path_factors.get(obj.path_relevance, 0.10)
        # Directly in path if corridor_overlap >= 0.50
        if obj.corridor_overlap >= 0.50:
            path_factor = cfg.path_factors.get("directly in path", 1.00)
        elif obj.corridor_overlap >= 0.20:
            path_factor = cfg.path_factors.get("partially in path", 0.70)
        elif abs(obj.horizontal_position) <= 0.35:
            path_factor = cfg.path_factors.get("near path", 0.40)
        else:
            path_factor = cfg.path_factors.get("outside path", 0.10)

        # 3. Confidence factor (clamp minimum to prevent suppressing obvious hazards)
        conf_factor = max(0.50, min(1.0, float(obj.confidence)))

        # 4. Persistence factor based on frames seen
        frames = getattr(obj, "frames_seen", 1)
        if frames >= 3:
            persistence = cfg.persistence_factors.get(3, 1.00)
        elif frames == 2:
            persistence = cfg.persistence_factors.get(2, 0.70)
        else:
            persistence = cfg.persistence_factors.get(1, 0.40)

        # Base compound risk formula
        # risk = criticality * distance_factor * path_factor * confidence * persistence
        base_risk = crit * dist_factor * path_factor * conf_factor * persistence

        # 5. Critical Overrides (bypass normal multiplication for immediate hazards)
        is_override = False
        lower_cls = obj.class_name.lower()
        is_center = obj.direction in ("center", "slightly left", "slightly right")
        is_very_near = obj.distance_category == "VERY_NEAR"
        is_near = obj.distance_category in ("VERY_NEAR", "NEAR")

        # A) Stairs + Center + VERY_NEAR
        if ("stair" in lower_cls or "step" in lower_cls) and is_center and is_very_near:
            is_override = True
            base_risk = max(base_risk, 0.95)

        # B) Possible drop / hole + Center + NEAR
        elif ("drop" in lower_cls or "hole" in lower_cls or "manhole" in lower_cls) and is_center and is_near:
            is_override = True
            base_risk = max(base_risk, 0.90)

        # C) Vehicle + Center + VERY_NEAR
        elif any(v in lower_cls for v in ["car", "bus", "truck", "motorcycle", "vehicle"]) and is_center and is_very_near:
            is_override = True
            base_risk = max(base_risk, 0.95)

        # D) Unknown obstacle + Center + VERY_NEAR
        elif "unidentified" in lower_cls and is_center and is_very_near:
            is_override = True
            base_risk = max(base_risk, 0.85)

        final_risk = round(float(min(1.0, max(0.0, base_risk))), 2)
        return crit, final_risk, is_override
