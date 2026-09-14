"""
Guidance Generator Module.
Generates concise, natural spoken guidance alerts using deterministic rule-based templates.
NO LLM is used.
Preserves: OBJECT + DIRECTION + DISTANCE CATEGORY (using "ahead" for center).
"""

from typing import Dict, Any

class GuidanceGenerator:
    """Generates natural language guidance alerts from structured hazard events."""

    def format_direction(self, direction_str: str) -> str:
        """Converts internal direction to natural speech phrasing."""
        d = direction_str.lower().strip()
        if d == "center":
            return "ahead"
        return d

    def format_distance_category(self, distance_cat: str) -> str:
        """Converts distance category to natural speech phrasing."""
        d = distance_cat.upper().strip()
        if d == "VERY_NEAR":
            return "very near"
        elif d == "NEAR":
            return "near"
        elif d == "MEDIUM":
            return "medium"
        elif d == "FAR":
            return "far"
        elif d == "VERY_FAR":
            return "very far"
        return d.lower()

    def format_object_name(self, class_name: str) -> str:
        """Capitalizes and cleans object name for spoken announcement."""
        cn = class_name.strip()
        if cn.lower() == "unidentified obstacle":
            return "Unidentified obstacle"
        elif cn.lower() == "possible drop":
            return "Possible drop"
        elif cn.lower() == "ground rise":
            return "Ground rise"
        return cn.capitalize()

    def generate(self, obj: Dict[str, Any]) -> str:
        """
        Builds standard guidance message:
        Example: "Person slightly left, near."
        For critical hazard: "Stop. Stairs ahead, very near."
        """
        cls_name = self.format_object_name(obj.get("class", obj.get("class_name", "Obstacle")))
        direction = self.format_direction(obj.get("direction", "center"))
        distance_cat = self.format_distance_category(obj.get("distance_category", "NEAR"))
        priority = obj.get("priority", "MEDIUM")

        # Critical life-safety prefix
        if priority == "CRITICAL" and distance_cat in ("very near", "near") and direction in ("ahead", "slightly left", "slightly right"):
            return f"Stop. {cls_name} {direction}, {distance_cat}."

        return f"{cls_name} {direction}, {distance_cat}."
