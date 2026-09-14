"""
Criticality Configuration & Lookup Module.
Assigns base hazard criticality scores to detected obstacle classes.
"""

from typing import Dict, Optional
from app.inference.config import RiskConfig, pipeline_config

class CriticalityTable:
    """Manages baseline hazard ratings for real-world obstacle classes."""
    def __init__(self, config: RiskConfig = None):
        self.config = config or pipeline_config.risk
        self.table: Dict[str, float] = dict(self.config.criticality_table)

    def get(self, class_name: str) -> float:
        """Retrieves criticality score for a given class name."""
        key = class_name.lower().strip()
        if key in self.table:
            return self.table[key]

        # Substring fuzzy match (e.g. "stairs_down" -> "stairs")
        for k, val in self.table.items():
            if k in key or key in k:
                return val

        return self.config.default_criticality

    def set_criticality(self, class_name: str, value: float) -> None:
        """Allows dynamic runtime tuning of class criticality."""
        self.table[class_name.lower().strip()] = float(min(1.0, max(0.0, value)))
