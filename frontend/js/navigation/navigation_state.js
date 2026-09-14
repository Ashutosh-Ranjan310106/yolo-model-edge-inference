/**
 * NavigationState: Synthesizes detected objects, fused depths, and sector hazard information
 * into a structured, real-time spatial scene representation.
 */

export class NavigationState {
  constructor() {
    this.latestState = null;
  }

  /**
   * Update and produce structured scene state
   */
  update(tracks, sectors, frameWidth, frameHeight) {
    const timestamp = Date.now();

    // 1. Identify closest object
    let closestObject = null;
    let minObjDist = Infinity;

    for (const obj of tracks) {
      if (obj.estimated_distance !== null && obj.estimated_distance < minObjDist) {
        minObjDist = obj.estimated_distance;
        closestObject = obj;
      }
    }

    // 2. Identify highest-priority hazard
    let primaryHazard = null;

    // Check center sector first
    if (sectors.center.has_obstacle) {
      primaryHazard = {
        type: "sector_obstacle",
        sector: "center",
        distance_m: sectors.center.min_distance_m,
        message: `Obstacle ${sectors.center.min_distance_m}m directly ahead`
      };
    } else if (closestObject && closestObject.estimated_distance <= 2.0) {
      primaryHazard = {
        type: "classified_object",
        sector: closestObject.sector,
        distance_m: closestObject.estimated_distance,
        message: `${closestObject.className} at ${closestObject.distance_label} in ${closestObject.sector}`
      };
    } else if (sectors.left.has_obstacle) {
      primaryHazard = {
        type: "sector_obstacle",
        sector: "left",
        distance_m: sectors.left.min_distance_m,
        message: `Obstacle ${sectors.left.min_distance_m}m to the Left`
      };
    } else if (sectors.right.has_obstacle) {
      primaryHazard = {
        type: "sector_obstacle",
        sector: "right",
        distance_m: sectors.right.min_distance_m,
        message: `Obstacle ${sectors.right.min_distance_m}m to the Right`
      };
    }

    // 3. Generate summary text for on-screen banner & future speech output
    let summaryText = "Path Clear";
    if (primaryHazard) {
      summaryText = `⚠️ ${primaryHazard.message}`;
    } else if (closestObject) {
      summaryText = `Detected: ${closestObject.className} (${closestObject.distance_label})`;
    }

    const state = {
      timestamp,
      objects: tracks.map(t => ({
        id: t.trackId,
        class_name: t.className,
        confidence: t.score,
        distance_m: t.estimated_distance,
        distance_label: t.distance_label,
        sector: t.sector,
        bbox: t.box
      })),
      sectors,
      primaryHazard,
      summaryText,
      totalObjects: tracks.length
    };

    this.latestState = state;
    return state;
  }
}
