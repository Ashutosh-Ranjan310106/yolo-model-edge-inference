/**
 * SpatialDepthFusion: Fuses 2D bounding boxes from YOLO with dense depth map,
 * calculates continuous horizontal position, 7-zone direction, 5 distance categories,
 * walking corridor trapezoid overlap, and 3-sector path clearance.
 */

export class SpatialDepthFusion {
  constructor() {
    this.obstacleThresholdMeters = 2.0;

    // Criticality weights for road navigation
    this.criticalityTable = {
      "manhole": 1.00,
      "stairs": 1.00,
      "vehicle": 0.95,
      "car": 0.95,
      "bus": 0.95,
      "truck": 0.95,
      "motorcycle": 0.90,
      "bike/motorcycle": 0.88,
      "bike": 0.85,
      "person": 0.70,
      "dog": 0.70,
      "animal": 0.70,
      "teraffic barrel": 0.75,
      "traffic barrel": 0.75,
      "guard rail": 0.70,
      "guardrail": 0.70,
      "electrical pole": 0.70,
      "pole": 0.70,
      "traffic sign": 0.60,
      "sign_board": 0.60,
      "traffic cone": 0.65,
      "cone": 0.65,
      "electrical box": 0.65,
      "fire hydrant": 0.60,
      "door": 0.55,
      "wall": 0.50,
      "building": 0.50,
      "window": 0.45,
      "bicycle rack": 0.45,
      "plant pot": 0.45,
      "table/desk": 0.40,
      "bookshelf/storage": 0.40,
      "display": 0.40,
      "dustbin": 0.40,
      "drawer": 0.35,
      "chair": 0.35,
      "bench": 0.30,
      "pedestrian crosswalk": 0.30,
      "road": 0.20,
      "unknown": 0.75
    };
  }

  classifyDirection(cx, imageWidth) {
    if (imageWidth <= 0) return { direction: "center", horizontalPosition: 0.0 };
    const halfW = imageWidth / 2.0;
    const relX = Math.max(-1.0, Math.min(1.0, (cx - halfW) / halfW));
    const hPos = Number(relX.toFixed(2));

    let direction = "center";
    if (relX < -0.70) direction = "far left";
    else if (relX < -0.35) direction = "left";
    else if (relX < -0.12) direction = "slightly left";
    else if (relX <= 0.12) direction = "center";
    else if (relX <= 0.35) direction = "slightly right";
    else if (relX <= 0.70) direction = "right";
    else direction = "far right";

    return { direction, horizontalPosition: hPos };
  }

  categorizeDistance(distM) {
    if (distM === null || distM === undefined) return "FAR";
    if (distM < 1.0) return "VERY_NEAR";
    if (distM < 2.0) return "NEAR";
    if (distM < 5.0) return "MEDIUM";
    if (distM <= 10.0) return "FAR";
    return "VERY_FAR";
  }

  computeCorridorOverlap(box, imageWidth, imageHeight) {
    const [x1, y1, x2, y2] = box;
    const boxW = Math.max(0, x2 - x1);
    const boxH = Math.max(0, y2 - y1);
    const boxArea = boxW * boxH;
    if (boxArea <= 0) return { overlap: 0.0, relevance: "outside path" };

    // Corridor trapezoid in image coordinates
    const topY = imageHeight * 0.45;
    const botY = imageHeight * 1.00;
    const cx = imageWidth / 2.0;
    const topHW = (imageWidth * 0.30) / 2.0;
    const botHW = (imageWidth * 0.80) / 2.0;

    // Check overlap via 5-point grid sampling
    let insidePoints = 0;
    const testPoints = [
      [x1 + boxW * 0.2, y1 + boxH * 0.2],
      [x1 + boxW * 0.8, y1 + boxH * 0.2],
      [x1 + boxW * 0.5, y1 + boxH * 0.5],
      [x1 + boxW * 0.2, y1 + boxH * 0.8],
      [x1 + boxW * 0.8, y1 + boxH * 0.8]
    ];

    for (const [px, py] of testPoints) {
      if (py >= topY && py <= botY) {
        const t = (py - topY) / (botY - topY);
        const currentHW = topHW + t * (botHW - topHW);
        if (px >= cx - currentHW && px <= cx + currentHW) {
          insidePoints++;
        }
      }
    }

    const overlapRatio = insidePoints / 5.0;
    let relevance = "outside path";
    if (overlapRatio >= 0.6) relevance = "direct path";
    else if (overlapRatio > 0.1) relevance = "near path";

    return { overlap: overlapRatio, relevance };
  }

  fuse(detections, depthModelInstance, imageWidth, imageHeight) {
    const fusedDetections = [];
    const hasDenseDepth = depthModelInstance && depthModelInstance.isLoaded;

    for (const det of detections) {
      const box = det.box;
      const [x1, y1, x2, y2] = box;
      const cx = (x1 + x2) / 2.0;

      const { direction, horizontalPosition } = this.classifyDirection(cx, imageWidth);
      const { overlap, relevance } = this.computeCorridorOverlap(box, imageWidth, imageHeight);

      let distM = det.distanceM;
      let distSource = "heuristic";

      if (hasDenseDepth) {
        const x1N = x1 / imageWidth;
        const y1N = y1 / imageHeight;
        const x2N = x2 / imageWidth;
        const y2N = y2 / imageHeight;

        const sampledDepth = depthModelInstance.getMedianDepthInROI(x1N, y1N, x2N, y2N);
        if (sampledDepth !== null && sampledDepth > 0) {
          distM = Number(sampledDepth.toFixed(2));
          distSource = "depth_map";
        }
      }

      const distCategory = this.categorizeDistance(distM);

      // Criticality & Risk Calculation
      const classKey = (det.className || "").toLowerCase();
      const baseCriticality = this.criticalityTable[classKey] !== undefined
        ? this.criticalityTable[classKey]
        : 0.50;

      let distWeight = 0.2;
      if (distM < 1.0) distWeight = 1.0;
      else if (distM < 2.0) distWeight = 0.8;
      else if (distM < 4.0) distWeight = 0.5;
      else if (distM < 7.0) distWeight = 0.3;

      let corridorWeight = 0.3;
      if (relevance === "direct path") corridorWeight = 1.0;
      else if (relevance === "near path") corridorWeight = 0.6;

      const riskScore = Number((baseCriticality * 0.4 + distWeight * 0.4 + corridorWeight * 0.2).toFixed(2));

      let priority = "LOW";
      if (distM < 1.2 && relevance === "direct path") priority = "CRITICAL";
      else if (distM < 2.0 && relevance !== "outside path") priority = "HIGH";
      else if (riskScore > 0.60) priority = "HIGH";
      else if (riskScore > 0.40) priority = "MEDIUM";
      else if (distM > 8.0) priority = "IGNORE";

      fusedDetections.push({
        ...det,
        distanceM: distM,
        distanceSource: distSource,
        distanceCategory: distCategory,
        direction,
        horizontalPosition,
        corridorOverlap: overlap,
        relevance,
        risk: riskScore,
        priority
      });
    }

    // Sort by risk descending
    fusedDetections.sort((a, b) => b.risk - a.risk);

    // Analyze 3-Sector Clearance (Left, Center, Right)
    const sectors = this.evaluateSectors(fusedDetections, hasDenseDepth, depthModelInstance);

    return {
      fusedDetections,
      sectors
    };
  }

  evaluateSectors(fusedDetections, hasDenseDepth, depthModelInstance) {
    const sectors = {
      left: { label: "Clear", distanceM: null, hasObstacle: false },
      center: { label: "Clear", distanceM: null, hasObstacle: false },
      right: { label: "Clear", distanceM: null, hasObstacle: false }
    };

    for (const d of fusedDetections) {
      if (d.distanceM === null || d.distanceM > 4.0) continue;

      let sec = "center";
      if (d.horizontalPosition < -0.30) sec = "left";
      else if (d.horizontalPosition > 0.30) sec = "right";

      if (!sectors[sec].distanceM || d.distanceM < sectors[sec].distanceM) {
        sectors[sec].distanceM = d.distanceM;
        sectors[sec].label = `${d.distanceM.toFixed(1)}m`;
        if (d.distanceM < this.obstacleThresholdMeters) {
          sectors[sec].hasObstacle = true;
        }
      }
    }

    return sectors;
  }
}
