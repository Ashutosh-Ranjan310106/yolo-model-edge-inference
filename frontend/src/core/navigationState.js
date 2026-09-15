/**
 * Navigation State & Voice Guidance Engine.
 * Evaluates current scene tracks, computes path clearance, and speaks
 * high-priority navigation instructions using the Web Speech Synthesis API.
 */

export class NavigationState {
  constructor() {
    this.primaryHazard = null;
    this.summaryText = "Path Clear";
    this.lastSpokenText = "";
    this.lastSpokenTime = 0;
    this.speechDebounceMs = 2500;
    this.speechSynthesis = typeof window !== "undefined" ? window.speechSynthesis : null;
    this.audioEnabled = true;
  }

  setAudioEnabled(enabled) {
    this.audioEnabled = enabled;
    if (!enabled && this.speechSynthesis) {
      this.speechSynthesis.cancel();
    }
  }

  update(tracks, sectors) {
    // Find highest risk object
    let topHazard = null;
    for (const t of tracks) {
      if (t.priority === "CRITICAL" || t.priority === "HIGH") {
        if (!topHazard || t.risk > topHazard.risk) {
          topHazard = t;
        }
      }
    }

    this.primaryHazard = topHazard;

    if (topHazard) {
      const distStr = topHazard.distanceM !== null ? `${topHazard.distanceM.toFixed(1)}m` : "ahead";
      const dirStr = topHazard.direction === "center" ? "directly ahead" : topHazard.direction;
      this.summaryText = `⚠️ ${topHazard.className} ${distStr} (${dirStr})`;

      // Debounced speech alert
      const spokenMsg = `${topHazard.className} ${distStr} ${dirStr}`;
      this.speak(spokenMsg, topHazard.priority);
    } else if (sectors && sectors.center && sectors.center.hasObstacle) {
      this.summaryText = `Obstacle ahead (~${sectors.center.label})`;
    } else {
      this.summaryText = "Path Clear";
    }

    return {
      summaryText: this.summaryText,
      primaryHazard: this.primaryHazard,
      sectors
    };
  }

  speak(text, priority = "LOW") {
    if (!this.audioEnabled || !this.speechSynthesis) return;

    const now = Date.now();
    const isCritical = priority === "CRITICAL";
    const minInterval = isCritical ? 1500 : this.speechDebounceMs;

    if (text === this.lastSpokenText && now - this.lastSpokenTime < minInterval) {
      return;
    }

    if (isCritical) {
      this.speechSynthesis.cancel();
    } else if (this.speechSynthesis.speaking) {
      return;
    }

    try {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.1;
      utterance.pitch = isCritical ? 1.2 : 1.0;
      utterance.volume = 1.0;

      utterance.onend = () => {
        this.lastSpokenTime = Date.now();
      };

      this.lastSpokenText = text;
      this.lastSpokenTime = now;
      this.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn("[NavigationState] SpeechSynthesis error:", e);
    }
  }
}
