/**
 * Multi-Object Temporal Tracker:
 * - Assigns persistent track IDs using bounding-box IoU association
 * - Smooths bounding-box coordinates and distance values via EMA
 * - Retains lost tracks for up to 3 frames to eliminate detection flicker
 */

export class TemporalTracker {
  constructor() {
    this.tracks = [];
    this.nextTrackId = 1;
    this.iouThreshold = 0.35;
    this.maxLostFrames = 3;
    this.alpha = 0.65; // Smoothing factor (0.65 current, 0.35 history)
  }

  reset() {
    this.tracks = [];
    this.nextTrackId = 1;
  }

  update(currentDetections) {
    const matchedTrackIndices = new Set();
    const matchedDetIndices = new Set();

    // 1. Associate current detections with existing active tracks
    for (let d = 0; d < currentDetections.length; d++) {
      const det = currentDetections[d];
      let bestIoU = 0;
      let bestTrackIdx = -1;

      for (let t = 0; t < this.tracks.length; t++) {
        if (matchedTrackIndices.has(t)) continue;
        const track = this.tracks[t];

        // Class compatibility check
        if (track.className !== det.className) continue;

        const iou = this.calculateIoU(track.box, det.box);
        if (iou > this.iouThreshold && iou > bestIoU) {
          bestIoU = iou;
          bestTrackIdx = t;
        }
      }

      if (bestTrackIdx !== -1) {
        matchedTrackIndices.add(bestTrackIdx);
        matchedDetIndices.add(d);

        // Update track with smoothed box & distance
        const track = this.tracks[bestTrackIdx];
        track.box = this.smoothBox(track.box, det.box, this.alpha);

        if (det.distanceM !== null && det.distanceM !== undefined) {
          track.distanceM = track.distanceM !== null
            ? Number((this.alpha * det.distanceM + (1 - this.alpha) * track.distanceM).toFixed(2))
            : det.distanceM;
        }

        track.score = det.score;
        track.direction = det.direction;
        track.horizontalPosition = det.horizontalPosition;
        track.distanceCategory = det.distanceCategory;
        track.priority = det.priority;
        track.risk = det.risk;
        track.relevance = det.relevance;
        track.corridorOverlap = det.corridorOverlap;
        track.lostFrames = 0;
        track.hits++;
      }
    }

    // 2. Add unmatched detections as new tracks
    for (let d = 0; d < currentDetections.length; d++) {
      if (!matchedDetIndices.has(d)) {
        const det = currentDetections[d];
        this.tracks.push({
          id: this.nextTrackId++,
          box: [...det.box],
          className: det.className,
          classId: det.classId,
          score: det.score,
          distanceM: det.distanceM,
          distanceCategory: det.distanceCategory,
          direction: det.direction,
          horizontalPosition: det.horizontalPosition,
          priority: det.priority,
          risk: det.risk,
          relevance: det.relevance,
          corridorOverlap: det.corridorOverlap,
          lostFrames: 0,
          hits: 1
        });
      }
    }

    // 3. Increment lost frames for unmatched existing tracks
    for (let t = 0; t < this.tracks.length; t++) {
      if (!matchedTrackIndices.has(t)) {
        this.tracks[t].lostFrames++;
      }
    }

    // 4. Prune dead tracks
    this.tracks = this.tracks.filter((t) => t.lostFrames <= this.maxLostFrames);

    // Return active tracks
    return this.tracks.map((t) => ({ ...t }));
  }

  smoothBox(prevBox, newBox, alpha) {
    return [
      alpha * newBox[0] + (1 - alpha) * prevBox[0],
      alpha * newBox[1] + (1 - alpha) * prevBox[1],
      alpha * newBox[2] + (1 - alpha) * prevBox[2],
      alpha * newBox[3] + (1 - alpha) * prevBox[3]
    ];
  }

  calculateIoU(b1, b2) {
    const interX1 = Math.max(b1[0], b2[0]);
    const interY1 = Math.max(b1[1], b2[1]);
    const interX2 = Math.min(b1[2], b2[2]);
    const interY2 = Math.min(b1[3], b2[3]);

    const interW = Math.max(0, interX2 - interX1);
    const interH = Math.max(0, interY2 - interY1);
    const interArea = interW * interH;

    const area1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    const area2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    const unionArea = area1 + area2 - interArea;

    return unionArea > 0 ? interArea / unionArea : 0;
  }
}
