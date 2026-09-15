import React from "react";
import { Volume2, VolumeX } from "lucide-react";

export function NavigationBanner({ summaryText, isHazard, audioEnabled, onToggleAudio }) {
  return (
    <div className={`nav-banner ${isHazard ? "hazard" : ""}`}>
      <span className="nav-banner-text">{summaryText}</span>
      <button
        className="btn-audio-toggle"
        onClick={onToggleAudio}
        title={audioEnabled ? "Disable Voice Guidance" : "Enable Voice Guidance"}
      >
        {audioEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
      </button>
    </div>
  );
}
