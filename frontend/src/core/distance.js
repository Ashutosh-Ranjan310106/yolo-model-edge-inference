/**
 * Distance Estimator Heuristic:
 * Fallback estimation using bounding-box area ratio when dense depth is unavailable.
 */

export const DISTANCE_TIERS = {
  VERY_CLOSE: { minRatio: 0.35, label: "<1 m (~0.8m)", distanceM: 0.8, category: "VERY_NEAR", priority: "CRITICAL" },
  NEAR:       { minRatio: 0.20, label: "1–2 m (~1.5m)", distanceM: 1.5, category: "NEAR", priority: "HIGH" },
  MID:        { minRatio: 0.10, label: "3–5 m (~3.5m)", distanceM: 3.5, category: "MEDIUM", priority: "MEDIUM" },
  FAR:        { minRatio: 0.05, label: "6–10 m (~7.5m)", distanceM: 7.5, category: "FAR", priority: "LOW" },
  DISTANT:    { minRatio: 0.00, label: ">10 m", distanceM: 12.0, category: "VERY_FAR", priority: "IGNORE" }
};

export class DistanceEstimator {
  estimate(box, frameWidth, frameHeight) {
    const [x1, y1, x2, y2] = box;
    const boxWidth = Math.max(0, x2 - x1);
    const boxHeight = Math.max(0, y2 - y1);
    const boxArea = boxWidth * boxHeight;
    const frameArea = frameWidth * frameHeight;

    if (frameArea <= 0) return { ...DISTANCE_TIERS.DISTANT, areaRatio: 0 };

    const areaRatio = boxArea / frameArea;

    if (areaRatio > 0.35) return { ...DISTANCE_TIERS.VERY_CLOSE, areaRatio };
    if (areaRatio > 0.20) return { ...DISTANCE_TIERS.NEAR, areaRatio };
    if (areaRatio > 0.10) return { ...DISTANCE_TIERS.MID, areaRatio };
    if (areaRatio > 0.05) return { ...DISTANCE_TIERS.FAR, areaRatio };
    return { ...DISTANCE_TIERS.DISTANT, areaRatio };
  }
}

export const distanceEstimator = new DistanceEstimator();
