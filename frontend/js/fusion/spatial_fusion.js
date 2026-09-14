/**
 * SpatialDepthFusion: Fuses 2D bounding boxes from YOLO with dense depth map,
 * calculates continuous horizontal position, 7-zone direction, 5 distance categories,
 * and trapezoidal walking corridor overlap.
 */

export class SpatialDepthFusion {
  constructor() {
    this.obstacleThresholdMeters = 2.0;

    // Criticality table matching backend
    this.criticalityTable = {
      "manhole": 1.00,
      "stairs": 1.00,
      "car": 0.95,
      "bus": 0.95,
      "truck": 0.95,
      "motorcycle": 0.90,
      "bike": 0.85,
      "person": 0.70,
      "dog": 0.70,
      "guard rail": 0.70,
      "guardrail": 0.70,
      "electrical pole": 0.70,
      "pole": 0.70,
      "traffic sign": 0.60,
      "traffic cone": 0.65,
      "cone": 0.65,
      "door": 0.55,
      "wall": 0.50,
      "plant pot": 0.45,
      "chair": 0.35,
      "bench": 0.30,
      "unknown": 0.75,
      "unidentified obstacle": 0.75
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

    // Corridor trapezoid in image coords
    const topY = imageHeight * 0.45;
    const botY = imageHeight * 1.00;
    const cx = imageWidth / 2.0;
    const topHW = (imageWidth * 0.30) / 2.0;
    const botHW = (imageWidth * 0.80) / 2.0;

    // Approximate trapezoid intersection via sampled grid points
    let insidePoints = 0;
    const samples = 16;
    for (let sy = 0; sy < 4; sy++) {
      const py = y1 + (sy + 0.5) * (boxH / 4);
      if (py < topY || py > botY) continue;

      const progress = (py - topY) / (botY - topY);
      const curHW = topHW + progress * (botHW - topHW);
      const cLeft = cx - curHW;
      const cRight = cx + curHW;

      for (let sx = 0; sx < 4; sx++) {
        const px = x1 + (sx + 0.5) * (boxW / 4);
        if (px >= cLeft && px <= cRight) {
          insidePoints++;
        }
      }
    }

    const overlap = Number((insidePoints / samples).toFixed(2));
    let relevance = "outside path";
    if (overlap >= 0.50) relevance = "likely in path";
    else if (overlap >= 0.20) relevance = "partially relevant";

    return { overlap, relevance };
  }

  getCriticality(className) {
    const key = (className || "").toLowerCase().trim();
    if (this.criticalityTable[key] !== undefined) return this.criticalityTable[key];
    for (const [k, v] of Object.entries(this.criticalityTable)) {
      if (key.includes(k) || k.includes(key)) return v;
    }
    return 0.50;
  }

  evaluateRisk(obj) {
    const crit = this.getCriticality(obj.className);

    // Distance multiplier
    let distFactor = 0.20;
    if (obj.distance_category === "VERY_NEAR") distFactor = 1.00;
    else if (obj.distance_category === "NEAR") distFactor = 0.80;
    else if (obj.distance_category === "MEDIUM") distFactor = 0.50;
    else if (obj.distance_category === "FAR") distFactor = 0.20;
    else distFactor = 0.05;

    // Path multiplier
    let pathFactor = 0.10;
    if (obj.corridor_overlap >= 0.50) pathFactor = 1.00;
    else if (obj.corridor_overlap >= 0.20) pathFactor = 0.70;
    else if (Math.abs(obj.horizontal_position) <= 0.35) pathFactor = 0.40;

    const confFactor = Math.max(0.50, Math.min(1.0, obj.score || 0.8));
    const persistence = obj.frames_seen >= 3 ? 1.0 : (obj.frames_seen === 2 ? 0.7 : 0.4);

    let risk = crit * distFactor * pathFactor * confFactor * persistence;

    // Critical safety overrides
    let isOverride = false;
    const lowerCls = (obj.className || "").toLowerCase();
    const isCenter = ["center", "slightly left", "slightly right"].includes(obj.direction);
    const isVeryNear = obj.distance_category === "VERY_NEAR";

    if ((lowerCls.includes("stair") || lowerCls.includes("step")) && isCenter && isVeryNear) {
      isOverride = true;
      risk = Math.max(risk, 0.95);
    } else if ((lowerCls.includes("drop") || lowerCls.includes("hole") || lowerCls.includes("manhole")) && isCenter && (isVeryNear || obj.distance_category === "NEAR")) {
      isOverride = true;
      risk = Math.max(risk, 0.90);
    } else if (["car", "bus", "truck", "motorcycle"].some(v => lowerCls.includes(v)) && isCenter && isVeryNear) {
      isOverride = true;
      risk = Math.max(risk, 0.95);
    } else if (lowerCls.includes("unidentified") && isCenter && isVeryNear) {
      isOverride = true;
      risk = Math.max(risk, 0.85);
    }

    risk = Number(Math.min(1.0, Math.max(0.0, risk)).toFixed(2));

    let priority = "LOW";
    if (isOverride || risk >= 0.75) priority = "CRITICAL";
    else if (risk >= 0.50) priority = "HIGH";
    else if (risk >= 0.25) priority = "MEDIUM";
    else if (risk < 0.10) priority = "IGNORE";

    return { criticality: crit, risk, priority, isOverride };
  }

  /**
   * Fuses YOLO detections with dense depth map
   */
  fuse(detections, depthModel, frameWidth, frameHeight) {
    const fusedDetections = [];

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.box;
      const x1Norm = x1 / frameWidth;
      const y1Norm = y1 / frameHeight;
      const x2Norm = x2 / frameWidth;
      const y2Norm = y2 / frameHeight;
      const cx = (x1 + x2) / 2.0;

      let estimatedDistance = null;
      let depthConfidence = 0.50;

      if (depthModel && depthModel.isLoaded && depthModel.latestDepthMap) {
        const depthM = depthModel.getMedianDepthInROI(x1Norm, y1Norm, x2Norm, y2Norm);
        if (depthM !== null && !isNaN(depthM) && depthM > 0) {
          estimatedDistance = depthM;
          depthConfidence = 0.85;
        }
      }

      // Bounding box heuristic fallback
      if (estimatedDistance === null) {
        const boxArea = Math.max(0, (x2 - x1) * (y2 - y1));
        const frameArea = frameWidth * frameHeight;
        const ratio = frameArea > 0 ? boxArea / frameArea : 0;
        if (ratio >= 0.35) estimatedDistance = 0.8;
        else if (ratio >= 0.20) estimatedDistance = 1.5;
        else if (ratio >= 0.10) estimatedDistance = 3.0;
        else if (ratio >= 0.05) estimatedDistance = 7.0;
        else estimatedDistance = 12.0;
        depthConfidence = 0.35;
      }

      // 1. Direction & Continuous Horizontal Position
      const { direction, horizontalPosition } = this.classifyDirection(cx, frameWidth);

      // 2. Distance Category
      const distCategory = this.categorizeDistance(estimatedDistance);

      // 3. Corridor Overlap
      const { overlap, relevance } = this.computeCorridorOverlap(det.box, frameWidth, frameHeight);

      // 4. Wall filtering (suppress background walls)
      if ((det.className || "").toLowerCase().includes("wall")) {
        if (overlap < 0.35 && (estimatedDistance || 5.0) > 2.5) {
          continue; // Ignore non-blocking peripheral wall
        }
      }

      const fusedItem = {
        ...det,
        class: det.className,
        estimated_distance: estimatedDistance,
        distance_m: estimatedDistance,
        distance_category: distCategory,
        direction,
        horizontal_position: horizontalPosition,
        corridor_overlap: overlap,
        path_relevance: relevance,
        depth_confidence: depthConfidence,
        frames_seen: det.frames_seen || 1
      };

      // 5. Risk & Priority
      const { criticality, risk, priority } = this.evaluateRisk(fusedItem);
      fusedItem.criticality = criticality;
      fusedItem.risk = risk;
      fusedItem.priority = priority;

      fusedDetections.push(fusedItem);
    }

    // Sectors from depth
    const sectors = this.computeSectorHazards(depthModel);

    return {
      fusedDetections,
      sectors
    };
  }

  computeSectorHazards(depthModel) {
    const defaultSectors = {
      left: { min_distance_m: null, has_obstacle: false, label: "Clear" },
      center: { min_distance_m: null, has_obstacle: false, label: "Clear" },
      right: { min_distance_m: null, has_obstacle: false, label: "Clear" }
    };

    if (!depthModel || !depthModel.isLoaded || !depthModel.latestDepthMap) {
      return defaultSectors;
    }

    const w = depthModel.depthWidth;
    const h = depthModel.depthHeight;
    const map = depthModel.latestDepthMap;

    const yStart = Math.floor(h * 0.30);
    const yEnd = Math.floor(h * 0.85);

    const xLeft = Math.floor(w * 0.33);
    const xRight = Math.floor(w * 0.66);

    const leftVals = [];
    const centerVals = [];
    const rightVals = [];

    for (let y = yStart; y < yEnd; y += 2) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x += 2) {
        const d = map[rowOffset + x];
        if (d > 0.2 && d < 20.0) {
          if (x < xLeft) leftVals.push(d);
          else if (x < xRight) centerVals.push(d);
          else rightVals.push(d);
        }
      }
    }

    const getSectorMin = (vals) => {
      if (vals.length === 0) return null;
      vals.sort((a, b) => a - b);
      const idx = Math.floor(vals.length * 0.05);
      return Math.round(vals[idx] * 10) / 10;
    };

    const minL = getSectorMin(leftVals);
    const minC = getSectorMin(centerVals);
    const minR = getSectorMin(rightVals);

    const checkObs = (dist) => dist !== null && dist <= this.obstacleThresholdMeters;

    return {
      left: { min_distance_m: minL, has_obstacle: checkObs(minL), label: minL ? `${minL.toFixed(1)}m` : "Clear" },
      center: { min_distance_m: minC, has_obstacle: checkObs(minC), label: minC ? `${minC.toFixed(1)}m` : "Clear" },
      right: { min_distance_m: minR, has_obstacle: checkObs(minR), label: minR ? `${minR.toFixed(1)}m` : "Clear" }
    };
  }
}
