/**
 * Universal Detection & Corridor Overlay Renderer.
 * Draws walking corridor trapezoid, bounding boxes, priority color tags,
 * metric distances, and risk ratings onto the 1:1 canvas.
 */

export function renderDetections(ctx, width, height, detections) {
  ctx.clearRect(0, 0, width, height);

  // 1. Draw walking corridor trapezoid reference overlay
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const topY = height * 0.45;
  const botY = height * 1.0;
  const topHW = (width * 0.30) / 2.0;
  const botHW = (width * 0.80) / 2.0;
  const cx = width / 2.0;
  ctx.moveTo(cx - topHW, topY);
  ctx.lineTo(cx + topHW, topY);
  ctx.lineTo(cx + botHW, botY);
  ctx.lineTo(cx - botHW, botY);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();

  if (!detections || detections.length === 0) return;

  const baseUnit = Math.max(0.7, width / 480);
  const fontSize = Math.max(11, Math.round(13 * baseUnit));
  ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
  ctx.textBaseline = "top";

  const lineWidth = Math.max(2, Math.round(2.5 * baseUnit));
  const labelHeight = Math.round(22 * baseUnit);

  for (const det of detections) {
    const b = det.box || det.bbox;
    if (!b || b.length < 4) continue;
    const [x1, y1, x2, y2] = b;
    const boxW = Math.max(0, x2 - x1);
    const boxH = Math.max(0, y2 - y1);

    const className = det.className || "Object";
    const score = det.score !== undefined ? det.score : 0;
    const priority = det.priority || "LOW";
    const direction = det.direction || "center";
    const distCat = det.distanceCategory || "FAR";
    const distM = det.distanceM !== undefined ? det.distanceM : null;
    const risk = det.risk !== undefined ? det.risk : null;

    // Priority-driven color palette
    let strokeColor = "#10b981"; // LOW / default green
    let textColor = "#000000";
    if (priority === "CRITICAL") {
      strokeColor = "#ef4444"; // Vivid Red
      textColor = "#ffffff";
    } else if (priority === "HIGH") {
      strokeColor = "#f59e0b"; // Vibrant Amber
      textColor = "#000000";
    } else if (priority === "MEDIUM") {
      strokeColor = "#38bdf8"; // Cyan
      textColor = "#000000";
    } else if (priority === "IGNORE") {
      strokeColor = "#64748b"; // Muted Slate
      textColor = "#ffffff";
    }

    // Draw bounding box
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeColor;
    ctx.strokeRect(x1, y1, boxW, boxH);

    // Label: CLASS CONF% • DIR • CAT (DISTm) • Risk: R
    const distPart = distM !== null && distM !== undefined ? `${distCat} (${Number(distM).toFixed(1)}m)` : distCat;
    const riskPart = risk !== null ? ` • R:${Number(risk).toFixed(2)}` : "";
    const labelText = `${className.toUpperCase()} ${(score * 100).toFixed(0)}% • ${direction} • ${distPart}${riskPart}`;
    const textWidth = ctx.measureText(labelText).width;

    // Label background banner
    const bannerY = Math.max(0, y1 - labelHeight);
    ctx.fillStyle = strokeColor;
    ctx.fillRect(x1, bannerY, textWidth + 10, labelHeight);

    // Label text
    ctx.fillStyle = textColor;
    ctx.fillText(labelText, x1 + 5, bannerY + 4);
  }
}
