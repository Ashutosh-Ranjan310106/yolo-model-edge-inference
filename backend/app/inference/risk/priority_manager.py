"""
Priority Manager Module.
Classifies objects and safety events into standard priority tiers:
CRITICAL, HIGH, MEDIUM, LOW, IGNORE.
Sorts items by priority and risk so only the most relevant hazards are announced.
"""

from typing import List, Dict, Any
from app.inference.config import RiskConfig, pipeline_config
from app.inference.spatial.spatial_fusion import SpatialObject

class PriorityManager:
    """Manages priority tiers and urgency ranking."""
    def __init__(self, config: RiskConfig = None):
        self.config = config or pipeline_config.risk

    def classify_priority(self, risk: float, is_critical_override: bool = False) -> str:
        """Maps numerical risk to priority level."""
        if is_critical_override:
            return "CRITICAL"

        cfg = self.config
        if risk >= cfg.priority_critical_threshold:
            return "CRITICAL"
        elif risk >= cfg.priority_high_threshold:
            return "HIGH"
        elif risk >= cfg.priority_medium_threshold:
            return "MEDIUM"
        elif risk >= cfg.priority_low_threshold:
            return "LOW"
        else:
            return "IGNORE"

    def rank_objects(self, evaluated_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts evaluated objects by urgency:
        Priority tier (CRITICAL > HIGH > MEDIUM > LOW > IGNORE), then risk score descending.
        """
        priority_weights = {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "IGNORE": 1
        }

        def sort_key(item: Dict[str, Any]):
            p = item.get("priority", "LOW")
            p_weight = priority_weights.get(p, 0)
            r = item.get("risk", 0.0)
            return (p_weight, r)

        return sorted(evaluated_objects, key=sort_key, reverse=True)
