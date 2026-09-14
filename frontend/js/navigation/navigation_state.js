/**
 * NavigationState: Synthesizes detected objects, fused depths, and corridor hazard information
 * into a structured, real-time spatial scene representation with priority ranking and spoken guidance.
 * 
 * Rules:
 * - Deterministic templates: OBJECT + DIRECTION + DISTANCE CATEGORY (using "ahead" for center).
 * - Priority ranking: CRITICAL > HIGH > MEDIUM > LOW > IGNORE.
 * - Cooldown enforcement to eliminate rapid chatter and repetitive speech.
 * - Web Speech API synthesis for hands-free navigation.
 */

export class NavigationState {
  constructor() {
    this.latestState = null;
    this.speechEnabled = true;

    // Cooldown timers (milliseconds)
    this.criticalCooldownMs = 1500;
    this.directionChangeCooldownMs = 2000;
    this.distanceChangeCooldownMs = 2000;
    this.sameEventCooldownMs = 4000;
    this.globalCooldownMs = 1800;

    // Active state tracker: key -> { lastTime, direction, distCat, risk, priority }
    this.announcedStates = new Map();
    this.lastGlobalSpeechTime = 0;
  }

  formatDirection(dir) {
    const d = (dir || "center").toLowerCase().trim();
    if (d === "center") return "ahead";
    return d;
  }

  formatDistanceCategory(cat) {
    const c = (cat || "NEAR").toUpperCase().trim();
    if (c === "VERY_NEAR") return "very near";
    if (c === "NEAR") return "near";
    if (c === "MEDIUM") return "medium";
    if (c === "FAR") return "far";
    if (c === "VERY_FAR") return "very far";
    return c.toLowerCase();
  }

  formatObjectName(name) {
    const cn = (name || "Obstacle").trim();
    if (cn.toLowerCase() === "unidentified obstacle") return "Unidentified obstacle";
    if (cn.toLowerCase() === "possible drop") return "Possible drop";
    if (cn.toLowerCase() === "ground rise") return "Ground rise";
    return cn.charAt(0).toUpperCase() + cn.slice(1);
  }

  generateGuidanceMessage(obj) {
    const clsName = this.formatObjectName(obj.className || obj.class || "Obstacle");
    const direction = this.formatDirection(obj.direction || "center");
    const distanceCat = this.formatDistanceCategory(obj.distance_category || "NEAR");
    const priority = obj.priority || "MEDIUM";

    if (priority === "CRITICAL" && ["very near", "near"].includes(distanceCat) && ["ahead", "slightly left", "slightly right"].includes(direction)) {
      return `Stop. ${clsName} ${direction}, ${distanceCat}.`;
    }

    return `${clsName} ${direction}, ${distanceCat}.`;
  }

  shouldAnnounce(obj, now) {
    const priority = obj.priority || "LOW";
    if (priority === "LOW" || priority === "IGNORE") return false;

    // Throttle non-critical speech by global cooldown
    if (priority !== "CRITICAL" && (now - this.lastGlobalSpeechTime) < this.globalCooldownMs) {
      return false;
    }

    const key = String(obj.id || obj.trackId || obj.className || "unknown");
    const prev = this.announcedStates.get(key);

    if (!prev) {
      return true; // First time seeing this hazard
    }

    const elapsed = now - prev.lastTime;

    // Critical escalation
    if (priority === "CRITICAL" && (prev.priority !== "CRITICAL" || elapsed >= this.criticalCooldownMs)) {
      return true;
    }

    // Priority escalated
    const priorityOrder = { "IGNORE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4 };
    if ((priorityOrder[priority] || 0) > (priorityOrder[prev.priority] || 0)) {
      return true;
    }

    // Distance category changed
    if (obj.distance_category !== prev.distCat && elapsed >= this.distanceChangeCooldownMs) {
      return true;
    }

    // Direction shifted significantly
    if (obj.direction !== prev.direction && elapsed >= this.directionChangeCooldownMs) {
      return true;
    }

    // Periodic reminder for continuing high/critical hazard
    if ((priority === "CRITICAL" || priority === "HIGH") && elapsed >= this.sameEventCooldownMs) {
      return true;
    }

    return false;
  }

  speak(text, priority = "MEDIUM") {
    if (!this.speechEnabled || typeof window === "undefined" || !window.speechSynthesis) return;

    try {
      if (priority === "CRITICAL") {
        window.speechSynthesis.cancel(); // Interrupt existing chatter for emergency alerts
      } else if (window.speechSynthesis.speaking) {
        return; // Don't speak over current speech
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = priority === "CRITICAL" ? 1.15 : 1.0;
      window.speechSynthesis.speak(utterance);
      this.lastGlobalSpeechTime = performance.now();
    } catch (e) {
      console.warn("[NavigationState] Speech error:", e);
    }
  }

  /**
   * Update and produce structured scene state
   */
  update(tracks, sectors, frameWidth, frameHeight) {
    const now = performance.now();
    const timestamp = Date.now();

    // 1. Rank tracks by priority & risk
    const priorityOrder = { "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "IGNORE": 4 };
    const sortedTracks = [...tracks].sort((a, b) => {
      const pA = priorityOrder[a.priority] !== undefined ? priorityOrder[a.priority] : 3;
      const pB = priorityOrder[b.priority] !== undefined ? priorityOrder[b.priority] : 3;
      if (pA !== pB) return pA - pB;

      const rA = a.risk || 0;
      const rB = b.risk || 0;
      if (Math.abs(rA - rB) > 0.05) return rB - rA;

      const dA = a.estimated_distance || a.distance_m || 999;
      const dB = b.estimated_distance || b.distance_m || 999;
      return dA - dB;
    });

    // 2. Identify primary hazard
    let primaryHazard = null;
    let guidanceEvents = [];

    const topCandidate = sortedTracks.find(t => t.priority === "CRITICAL" || t.priority === "HIGH" || t.priority === "MEDIUM");

    if (topCandidate) {
      const msg = this.generateGuidanceMessage(topCandidate);
      primaryHazard = {
        type: "classified_object",
        id: topCandidate.id || topCandidate.trackId,
        priority: topCandidate.priority,
        risk: topCandidate.risk,
        direction: topCandidate.direction,
        distance_category: topCandidate.distance_category,
        distance_m: topCandidate.estimated_distance || topCandidate.distance_m,
        message: msg
      };

      if (this.shouldAnnounce(topCandidate, now)) {
        guidanceEvents.push({
          type: "guidance",
          priority: topCandidate.priority,
          message: msg,
          target_id: topCandidate.id || topCandidate.trackId
        });

        // Update announced state
        const key = String(topCandidate.id || topCandidate.trackId || topCandidate.className || "unknown");
        this.announcedStates.set(key, {
          lastTime: now,
          direction: topCandidate.direction,
          distCat: topCandidate.distance_category,
          risk: topCandidate.risk || 0,
          priority: topCandidate.priority
        });

        // Trigger spoken utterance
        this.speak(msg, topCandidate.priority);
      }
    } else if (sectors && sectors.center && sectors.center.has_obstacle) {
      const msg = `Obstacle ahead, near.`;
      primaryHazard = {
        type: "sector_obstacle",
        priority: "HIGH",
        risk: 0.70,
        direction: "center",
        distance_category: "NEAR",
        distance_m: sectors.center.min_distance_m,
        message: msg
      };

      if ((now - this.lastGlobalSpeechTime) >= this.sameEventCooldownMs) {
        this.speak(msg, "HIGH");
        this.lastGlobalSpeechTime = now;
      }
    }

    // Clean up stale announced states
    for (const [key, state] of this.announcedStates.entries()) {
      if ((now - state.lastTime) > 8000) {
        this.announcedStates.delete(key);
      }
    }

    // 3. Generate summary banner text
    let summaryText = "Path Clear";
    if (primaryHazard) {
      summaryText = primaryHazard.priority === "CRITICAL"
        ? `⚠️ ${primaryHazard.message}`
        : primaryHazard.message;
    }

    const state = {
      timestamp,
      objects: sortedTracks,
      sectors,
      primaryHazard,
      events: guidanceEvents,
      summaryText,
      totalObjects: sortedTracks.length
    };

    this.latestState = state;
    return state;
  }
}

