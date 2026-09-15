import React from "react";
import { 
  Compass, 
  FileText, 
  ArrowRight, 
  ExternalLink, 
  Zap, 
  ShieldCheck, 
  Sparkles, 
  Cpu, 
  Volume2, 
  Layers, 
  BrainCircuit,
  BookOpen
} from "lucide-react";

export function HomeScreen({ onNavigateToNav }) {
  return (
    <main className="screen home-screen">
      {/* Hero Section */}
      <section className="home-hero">
        <div className="home-pill">
          <Sparkles size={13} className="pill-icon text-cyan" />
          <span>Multimodal Assistive Ecosystem</span>
        </div>
        <h1 className="hero-title">
          vision<span className="hero-gradient">X</span>
        </h1>
        <p className="hero-subtitle">
          Next-Generation Edge Intelligence & Multimodal Perception
        </p>
        <p className="hero-description">
          Empowering real-time spatial awareness and intelligent document synthesis with 100% on-device neural processing and cloud-accelerated analytics.
        </p>
      </section>

      {/* Two Main Platform Cards */}
      <section className="home-cards-grid">
        {/* Card 1: VisionX Navigation */}
        <div className="platform-card nav-card">
          <div className="card-ambient-glow glow-cyan"></div>
          <div className="card-header">
            <div className="card-icon-box nav-icon-box">
              <Compass size={28} />
            </div>
            <div className="card-tags">
              <span className="badge badge-edge">
                <Cpu size={11} style={{ marginRight: 4 }} />
                100% ON-DEVICE
              </span>
              <span className="badge badge-subtle">WASM / WEBGPU</span>
            </div>
          </div>

          <div className="card-body">
            <h2 className="card-heading">VisionX Navigation</h2>
            <p className="card-tagline">
              Real-Time Spatial Perception & Auditory Corridor Guidance
            </p>
            <p className="card-text">
              Fully standalone on-device obstacle detection and dense 3D depth perception. Engineered for visually impaired navigation, hazard avoidance, and continuous auditory route feedback without server latency.
            </p>

            <ul className="feature-list">
              <li>
                <Zap size={14} className="feat-icon text-amber" />
                <span><strong>Dual YOLO26 & Depth Engine:</strong> Real-time 25+ obstacle detection fused with metric depth.</span>
              </li>
              <li>
                <Layers size={14} className="feat-icon text-cyan" />
                <span><strong>3D Corridor Tracking:</strong> 7-zone direction analysis & walking path safety checks.</span>
              </li>
              <li>
                <Volume2 size={14} className="feat-icon text-emerald" />
                <span><strong>Continuous Voice Guidance:</strong> Natural spatial sound cues for immediate hazard awareness.</span>
              </li>
              <li>
                <ShieldCheck size={14} className="feat-icon text-blue" />
                <span><strong>Zero Cloud Transmission:</strong> Camera stream never leaves local memory.</span>
              </li>
            </ul>
          </div>

          <div className="card-footer">
            <button className="btn btn-primary btn-launch" onClick={onNavigateToNav}>
              <span>Launch Navigation System</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>

        {/* Card 2: Handwritten Notes Analyser */}
        <div className="platform-card notes-card">
          <div className="card-ambient-glow glow-purple"></div>
          <div className="card-header">
            <div className="card-icon-box notes-icon-box">
              <FileText size={28} />
            </div>
            <div className="card-tags">
              <span className="badge badge-cloud">
                <Sparkles size={11} style={{ marginRight: 4 }} />
                AI MULTIMODAL
              </span>
              <span className="badge badge-subtle">OCR & SYNTHESIS</span>
            </div>
          </div>

          <div className="card-body">
            <h2 className="card-heading">Handwritten Notes Analyser</h2>
            <p className="card-tagline">
              Intelligent Document Digitization, Math OCR & Synthesis
            </p>
            <p className="card-text">
              Transform handwritten pages, whiteboard captures, mathematical equations, and lecture notes into structured digital insights, interactive quizzes, and concise concept breakdowns in seconds.
            </p>

            <ul className="feature-list">
              <li>
                <BrainCircuit size={14} className="feat-icon text-purple" />
                <span><strong>Neural Handwriting OCR:</strong> Precision transcription of cursive and unconstrained handwriting.</span>
              </li>
              <li>
                <BookOpen size={14} className="feat-icon text-cyan" />
                <span><strong>Formula & Diagram Parsing:</strong> Mathematical symbols, chemical diagrams, and structured notes.</span>
              </li>
              <li>
                <Sparkles size={14} className="feat-icon text-amber" />
                <span><strong>Instant Executive Summaries:</strong> Key takeaway extraction, flashcards, and study aids.</span>
              </li>
              <li>
                <ExternalLink size={14} className="feat-icon text-emerald" />
                <span><strong>Accessible Anywhere:</strong> Cloud-accelerated web interface for rapid document uploads.</span>
              </li>
            </ul>
          </div>

          <div className="card-footer">
            <a 
              href="https://handwritten-notes-analyser-sgwn.vercel.app/" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="btn btn-notes-action"
            >
              <span>Open Notes Analyser</span>
              <ExternalLink size={16} />
            </a>
          </div>
        </div>
      </section>

      {/* Platform Capabilities Stats Footer */}
      <footer className="home-footer">
        <div className="footer-stat">
          <span className="stat-val">100%</span>
          <span className="stat-label">Privacy First</span>
        </div>
        <div className="footer-divider"></div>
        <div className="footer-stat">
          <span className="stat-val">&lt; 150ms</span>
          <span className="stat-label">Local Edge Latency</span>
        </div>
        <div className="footer-divider"></div>
        <div className="footer-stat">
          <span className="stat-val">Dual AI</span>
          <span className="stat-label">Vision & Notes Suite</span>
        </div>
      </footer>
    </main>
  );
}
