"""
Event & Announcement Manager Module.
Prevents rapid chatter and repeated speech.
Re-announces only on significant state changes:
- New high-relevance object appears
- Direction shifts significantly
- Distance category changes
- Risk increases significantly
- Priority escalates
- Critical hazard appears
Enforces configurable cooldown timers.
"""

from typing import List, Dict, Any, Optional
import time
from app.inference.config import GuidanceConfig, pipeline_config
from app.inference.guidance.guidance_generator import GuidanceGenerator

class EventManager:
    """Manages guidance event dispatching, state transition detection, and cooldowns."""
    def __init__(self, config: GuidanceConfig = None):
        self.config = config or pipeline_config.guidance
        self.generator = GuidanceGenerator()
        # Active announcement states: { track_id_or_key: { "last_time": float, "direction": str, "dist_cat": str, "risk": float, "priority": str } }
        self.announced_states: Dict[str, Dict[str, Any]] = {}
        self.last_global_announcement_time: float = 0.0

    def should_announce(self, obj: Dict[str, Any], now: float) -> Tuple[bool, str]:
        """
        Determines whether an object should trigger an announcement.
        Returns (should_announce, reason).
        """
        cfg = self.config
        obj_key = str(obj.get("id") or obj.get("class", "unknown"))
        priority = obj.get("priority", "LOW")
        direction = obj.get("direction", "center")
        dist_cat = obj.get("distance_category", "NEAR")
        risk = obj.get("risk", 0.0)

        # Ignore LOW and IGNORE events
        if priority in ("LOW", "IGNORE"):
            return False, "low_priority"

        prev = self.announced_states.get(obj_key)

        # 1. First time seeing this object at relevant priority
        if prev is None:
            return True, "new_object"

        elapsed = now - prev["last_time"]

        # 2. Critical hazard escalation: instant announcement with minimal cooldown
        if priority == "CRITICAL" and (prev["priority"] != "CRITICAL" or elapsed >= cfg.critical_hazard_cooldown_sec):
            return True, "critical_escalation"

        # 3. Priority escalation (e.g. MEDIUM -> HIGH)
        if priority != prev["priority"]:
            priority_order = {"IGNORE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            if priority_order.get(priority, 0) > priority_order.get(prev["priority"], 0):
                return True, "priority_escalated"

        # 4. Distance category changed (e.g. MEDIUM -> NEAR)
        if dist_cat != prev["dist_cat"] and elapsed >= cfg.distance_change_cooldown_sec:
            return True, "distance_changed"

        # 5. Direction changed significantly
        if direction != prev["direction"] and elapsed >= cfg.direction_change_cooldown_sec:
            return True, "direction_changed"

        # 6. Risk increased significantly (+0.25)
        if (risk - prev["risk"]) >= 0.25 and elapsed >= cfg.same_event_cooldown_sec:
            return True, "risk_increased"

        # 7. Same state cooldown expired for HIGH/CRITICAL objects still persisting
        if priority in ("CRITICAL", "HIGH") and elapsed >= cfg.same_event_cooldown_sec * 2:
            return True, "periodic_critical_reminder"

        return False, "cooldown_active"

    def process(self, ranked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluates ranked objects and dispatches guidance events without repetitive speech.
        Returns list of GuidanceEvent dicts: [{ "type": "guidance", "priority": ..., "message": ... }]
        """
        now = time.time()
        events: List[Dict[str, Any]] = []

        # Only evaluate top candidates to avoid flooding user
        top_candidates = [o for o in ranked_objects if o.get("priority") in ("CRITICAL", "HIGH", "MEDIUM")]

        for obj in top_candidates:
            ok, reason = self.should_announce(obj, now)
            if ok:
                msg = self.generator.generate(obj)
                event = {
                    "type": "guidance",
                    "priority": obj.get("priority", "MEDIUM"),
                    "message": msg,
                    "target_id": obj.get("id"),
                    "class": obj.get("class"),
                    "direction": obj.get("direction"),
                    "distance_category": obj.get("distance_category"),
                    "reason": reason
                }
                events.append(event)

                # Record announcement state
                obj_key = str(obj.get("id") or obj.get("class", "unknown"))
                self.announced_states[obj_key] = {
                    "last_time": now,
                    "direction": obj.get("direction"),
                    "dist_cat": obj.get("distance_category"),
                    "risk": obj.get("risk", 0.0),
                    "priority": obj.get("priority")
                }
                self.last_global_announcement_time = now

                # Stop after emitting max_concurrent_announcements (default 1) to keep speech clean
                if len(events) >= self.config.max_concurrent_announcements:
                    break

        # Prune ancient announced states (> 10 seconds without seeing)
        stale_keys = [k for k, v in self.announced_states.items() if (now - v["last_time"]) > 10.0]
        for k in stale_keys:
            del self.announced_states[k]

        return events
