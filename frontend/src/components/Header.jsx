import React from "react";
import { Cpu, Home } from "lucide-react";

export function Header({ currentScreen = "home", onGoHome, cachedCount = 0 }) {
  const isHome = currentScreen === "home";

  return (
    <header className="app-header">
      <div 
        className={`brand ${!isHome ? "brand-interactive" : ""}`}
        onClick={!isHome && onGoHome ? onGoHome : undefined}
        title={!isHome ? "Return to VisionX Home" : "VisionX Platform"}
        role={!isHome ? "button" : undefined}
      >
        <span className="brand-icon">🧭</span>
        <h1>
          vision<span className="brand-accent">X</span>
          {!isHome && <span className="brand-subpage"> / Navigation</span>}
        </h1>
        <span className="badge">
          <Cpu size={12} style={{ marginRight: 4 }} />
          ON-DEVICE
        </span>
      </div>

      <div className="header-right">
        {!isHome && onGoHome && (
          <button 
            className="btn-header-home" 
            onClick={onGoHome}
            title="Return to VisionX Home"
          >
            <Home size={13} style={{ marginRight: 5 }} />
            <span>Home</span>
          </button>
        )}
        <div className="system-status">
          <span className="status-dot online"></span>
          <span>{isHome ? "Platform Ready" : cachedCount > 0 ? `(${cachedCount} Cached)` : "(Ready)"}</span>
        </div>
      </div>
    </header>
  );
}
