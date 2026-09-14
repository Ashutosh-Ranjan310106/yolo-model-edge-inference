/**
 * TemporalTracker: Lightweight temporal tracking & distance smoothing for visual stability.
 * Prevents flicker, smooths noisy distance estimates (EMA), and provides track persistence.
 */

export class TemporalTracker {
  constructor() {
    this.tracks = []; // [{ id, classId, className, box, score, distance, lastSeen }]
    this.nextTrackId = 1;
    this.trackTimeoutMs = 600; // Keep track alive for up to 600ms across async inference gaps
    this.iouThreshold = 0.25;  // Tolerant IoU threshold for mobile hand-held camera movement
    this.alphaDist = 0.35; // Smoothing factor for distance (0..1, lower = smoother)
    this.alphaBox = 0.50;  // Smoothing factor for bounding box coordinates
  }

  computeIoU(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (interArea === 0) return 0;

    const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    return interArea / (boxAArea + boxBArea - interArea);
  }

  update(incomingDetections) {
    const now = performance.now();
    const unmatchedIncoming = new Set(incomingDetections.map((_, idx) => idx));
    const matchedTrackIds = new Set();

    // 1. Match existing tracks to incoming detections
    for (const track of this.tracks) {
      let bestIoU = 0;
      let bestIdx = -1;

      for (const idx of unmatchedIncoming) {
        const incoming = incomingDetections[idx];
        if (incoming.classId === track.classId) {
          const iou = this.computeIoU(track.box, incoming.box);
          if (iou > this.iouThreshold && iou > bestIoU) {
            bestIoU = iou;
            bestIdx = idx;
          }
        }
      }

      if (bestIdx !== -1) {
        const matched = incomingDetections[bestIdx];
        unmatchedIncoming.delete(bestIdx);
        matchedTrackIds.add(track.id);

        // Smooth box
        for (let k = 0; k < 4; k++) {
          track.box[k] = track.box[k] * (1.0 - this.alphaBox) + matched.box[k] * this.alphaBox;
        }

        // Smooth distance
        if (matched.estimated_distance !== null) {
          if (track.distance !== null) {
            track.distance = track.distance * (1.0 - this.alphaDist) + matched.estimated_distance * this.alphaDist;
          } else {
            track.distance = matched.estimated_distance;
          }
          track.distance_label = `${track.distance.toFixed(1)} m`;
        }

        track.score = matched.score;
        track.sector = matched.sector;
        track.distance_source = matched.distance_source;
        track.lastSeen = now;
      }
    }

    // 2. Add new tracks for unmatched detections
    for (const idx of unmatchedIncoming) {
      const inc = incomingDetections[idx];
      this.tracks.push({
        id: this.nextTrackId++,
        classId: inc.classId,
        className: inc.className,
        box: [...inc.box],
        score: inc.score,
        distance: inc.estimated_distance,
        distance_label: inc.distance_label,
        distance_source: inc.distance_source,
        sector: inc.sector,
        lastSeen: now
      });
    }

    // 3. Prune tracks that haven't been seen within trackTimeoutMs
    this.tracks = this.tracks.filter(t => (now - t.lastSeen) <= this.trackTimeoutMs);

    // 4. Return active stabilized detections
    return this.tracks.map(t => ({
      trackId: t.id,
      classId: t.classId,
      className: t.className,
      box: t.box.map(v => Math.round(v)),
      score: t.score,
      estimated_distance: t.distance !== null ? Math.round(t.distance * 10) / 10 : null,
      distance_label: t.distance !== null ? `${(Math.round(t.distance * 10) / 10).toFixed(1)} m` : t.distance_label,
      distance_source: t.distance_source,
      sector: t.sector,
      isInterpolated: (now - t.lastSeen) > 120
    }));
  }

  reset() {
    this.tracks = [];
    this.nextTrackId = 1;
  }
}
