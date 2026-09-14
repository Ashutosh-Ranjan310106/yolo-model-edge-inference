/**
 * SpatialDepthFusion: Fuses 2D bounding boxes from YOLO with dense per-pixel depth maps.
 * Calculates robust median distance per object and detects non-YOLO sector obstacles.
 */

export class SpatialDepthFusion {
  constructor() {
    this.obstacleThresholdMeters = 2.0; // Distance below which a surface is marked as an obstacle
  }

  /**
   * Fuses YOLO detections with dense depth map
   * @param {Array} detections - YOLO bounding box objects
   * @param {DepthModel} depthModel - DepthModel instance
   * @param {number} frameWidth - Live viewfinder width
   * @param {number} frameHeight - Live viewfinder height
   */
  fuse(detections, depthModel, frameWidth, frameHeight) {
    const fusedDetections = [];

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.box;
      const x1Norm = x1 / frameWidth;
      const y1Norm = y1 / frameHeight;
      const x2Norm = x2 / frameWidth;
      const y2Norm = y2 / frameHeight;

      let estimatedDistance = null;
      let distanceLabel = "";
      let source = "none";

      if (depthModel && depthModel.isLoaded && depthModel.latestDepthMap) {
        // Robust core sampling (inner 50% box median)
        const depthM = depthModel.getMedianDepthInROI(x1Norm, y1Norm, x2Norm, y2Norm);
        if (depthM !== null && !isNaN(depthM) && depthM > 0) {
          estimatedDistance = depthM;
          distanceLabel = `${depthM.toFixed(1)} m`;
          source = depthModel.isMetric ? "dense_metric_depth" : "dense_relative_depth";
        }
      }

      // Fallback to bounding-box area ratio heuristic if depth is unavailable
      if (estimatedDistance === null) {
        const boxArea = Math.max(0, (x2 - x1) * (y2 - y1));
        const frameArea = frameWidth * frameHeight;
        const ratio = frameArea > 0 ? boxArea / frameArea : 0;
        
        if (ratio >= 0.35) {
          estimatedDistance = 0.8;
          distanceLabel = "<1 m";
        } else if (ratio >= 0.20) {
          estimatedDistance = 1.5;
          distanceLabel = "1–2 m";
        } else if (ratio >= 0.10) {
          estimatedDistance = 3.0;
          distanceLabel = "3–5 m";
        } else if (ratio >= 0.05) {
          estimatedDistance = 7.0;
          distanceLabel = "6–10 m";
        } else {
          estimatedDistance = 12.0;
          distanceLabel = ">10 m";
        }
        source = "bbox_heuristic";
      }

      // Determine horizontal sector for object (left, center, right)
      const cxNorm = (x1Norm + x2Norm) / 2;
      let sector = "center";
      if (cxNorm < 0.33) sector = "left";
      else if (cxNorm > 0.66) sector = "right";

      fusedDetections.push({
        ...det,
        estimated_distance: estimatedDistance,
        distance_label: distanceLabel,
        distance_source: source,
        sector
      });
    }

    // Compute Depth-Only Sector Hazards (unclassified walls, surfaces, obstacles)
    const sectors = this.computeSectorHazards(depthModel);

    return {
      fusedDetections,
      sectors
    };
  }

  /**
   * Scans depth map across 3 horizontal sectors (Left, Center, Right)
   * in the central vertical band (25% to 80% height).
   */
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

    const yStart = Math.floor(h * 0.25);
    const yEnd = Math.floor(h * 0.80);

    const xLeft = Math.floor(w * 0.33);
    const xRight = Math.floor(w * 0.66);

    const leftVals = [];
    const centerVals = [];
    const rightVals = [];

    // Step by 2 pixels for fast mobile processing
    for (let y = yStart; y < yEnd; y += 2) {
      const rowOffset = y * w;
      for (let x = 0; x < w; x += 2) {
        const d = map[rowOffset + x];
        if (d > 0.2 && d < 25.0) {
          if (x < xLeft) {
            leftVals.push(d);
          } else if (x < xRight) {
            centerVals.push(d);
          } else {
            rightVals.push(d);
          }
        }
      }
    }

    const getSectorMin = (vals) => {
      if (vals.length === 0) return null;
      vals.sort((a, b) => a - b);
      // 5th percentile to discard outliers
      const idx = Math.floor(vals.length * 0.05);
      return Math.round(vals[idx] * 10) / 10;
    };

    const minL = getSectorMin(leftVals);
    const minC = getSectorMin(centerVals);
    const minR = getSectorMin(rightVals);

    const checkObs = (dist) => dist !== null && dist <= this.obstacleThresholdMeters;

    return {
      left: {
        min_distance_m: minL,
        has_obstacle: checkObs(minL),
        label: minL ? `${minL.toFixed(1)}m` : "Clear"
      },
      center: {
        min_distance_m: minC,
        has_obstacle: checkObs(minC),
        label: minC ? `${minC.toFixed(1)}m` : "Clear"
      },
      right: {
        min_distance_m: minR,
        has_obstacle: checkObs(minR),
        label: minR ? `${minR.toFixed(1)}m` : "Clear"
      }
    };
  }
}
