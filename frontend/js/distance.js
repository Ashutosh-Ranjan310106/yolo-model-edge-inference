/**
 * Distance Estimator heuristic based on bounding-box area ratio.
 * area_ratio = (bbox_width * bbox_height) / (frame_width * frame_height)
 * 
 * Distance Tiers:
 * > 35%       -> < 1 m
 * 20% - 35%   -> 1–2 m
 * 10% - 20%   -> 3–5 m
 * 5% - 10%    -> 6–10 m
 * < 5%        -> > 10 m
 */
export class DistanceEstimator {
  constructor(options = {}) {
    this.tiers = options.tiers || [
      { minRatio: 0.35, maxRatio: 1.00, label: "<1 m", approxMeters: 0.8 },
      { minRatio: 0.20, maxRatio: 0.35, label: "1–2 m", approxMeters: 1.5 },
      { minRatio: 0.10, maxRatio: 0.20, label: "3–5 m", approxMeters: 3.5 },
      { minRatio: 0.05, maxRatio: 0.10, label: "6–10 m", approxMeters: 7.5 },
      { minRatio: 0.00, maxRatio: 0.05, label: ">10 m", approxMeters: 12.0 }
    ];
  }

  /**
   * Estimate distance from box [x1, y1, x2, y2] and frame [frameWidth, frameHeight]
   * @param {number[]} box - [x1, y1, x2, y2]
   * @param {number} frameWidth
   * @param {number} frameHeight
   * @returns {{ label: string, meters: number, ratio: number }}
   */
  estimate(box, frameWidth, frameHeight) {
    const w = Math.max(0, box[2] - box[0]);
    const h = Math.max(0, box[3] - box[1]);
    const bboxArea = w * h;
    const frameArea = Math.max(1, frameWidth * frameHeight);
    const ratio = Math.min(1.0, bboxArea / frameArea);

    for (const tier of this.tiers) {
      if (ratio >= tier.minRatio && ratio <= tier.maxRatio) {
        return {
          label: tier.label,
          meters: tier.approxMeters,
          ratio: ratio,
          display: `~${tier.approxMeters}m`
        };
      }
    }

    return {
      label: ">10 m",
      meters: 12.0,
      ratio: ratio,
      display: ">10m"
    };
  }
}
