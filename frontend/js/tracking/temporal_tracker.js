/**
 * TemporalTracker: Lightweight temporal tracking & distance smoothing for visual stability.
 * Prevents flicker, smooths noisy distance estimates (EMA), tracks horizontal position,
 * and maintains track persistence and hazard rankings across frames.
 */

export class TemporalTracker {
  constructor() {
    this.tracks = []; // [{ id, classId, className, box, score, distance, direction, horizontal_position, ... }]
    this.nextTrackId = 1;
    this.trackTimeoutMs = 600; // Keep track alive for up to 600ms across async inference gaps
    this.iouThreshold = 0.25;  // Tolerant IoU threshold for mobile hand-held camera movement
    this.alphaDist = 0.35;     // Smoothing factor for distance (0..1, lower = smoother)
    this.alphaBox = 0.50;      // Smoothing factor for bounding box coordinates
    this.alphaPos = 0.50;      // Smoothing factor for horizontal normalized position
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

    // 1. Match existing tracks to incoming detections via IoU and class compatibility
    for (const track of this.tracks) {
      let bestIoU = 0;
      let bestIdx = -1;

      for (const idx of unmatchedIncoming) {
        const incoming = incomingDetections[idx];
        const incomingClass = incoming.className || incoming.class || "";
        const trackClass = track.className || track.class || "";

        if (incoming.classId === track.classId || incomingClass === trackClass) {
          const iou = this.computeIoU(track.box, incoming.box || incoming.bbox);
          if (iou > this.iouThreshold && iou > bestIoU) {
            bestIoU = iou;
            bestIdx = idx;
          }
        }
      }

      if (bestIdx !== -1) {
        const matched = incomingDetections[bestIdx];
        const matchedBox = matched.box || matched.bbox;
        unmatchedIncoming.delete(bestIdx);
        matchedTrackIds.add(track.id);

        // Smooth bounding box
        for (let k = 0; k < 4; k++) {
          track.box[k] = track.box[k] * (1.0 - this.alphaBox) + matchedBox[k] * this.alphaBox;
        }

        // Smooth distance (EMA)
        const inDist = matched.estimated_distance !== undefined ? matched.estimated_distance : matched.distance_m;
        if (inDist !== null && inDist !== undefined && !isNaN(inDist)) {
          if (track.distance !== null && !isNaN(track.distance)) {
            track.distance = track.distance * (1.0 - this.alphaDist) + inDist * this.alphaDist;
          } else {
            track.distance = inDist;
          }
          track.distance_label = `${track.distance.toFixed(1)} m`;
        }

        // Smooth continuous horizontal position
        if (matched.horizontal_position !== undefined && matched.horizontal_position !== null) {
          track.horizontal_position = Number(
            (track.horizontal_position * (1.0 - this.alphaPos) + matched.horizontal_position * this.alphaPos).toFixed(2)
          );
        }

        track.score = matched.score !== undefined ? matched.score : matched.confidence;
        track.direction = matched.direction || track.direction;
        track.distance_category = matched.distance_category || track.distance_category;
        track.corridor_overlap = matched.corridor_overlap !== undefined ? matched.corridor_overlap : track.corridor_overlap;
        track.path_relevance = matched.path_relevance || track.path_relevance;
        track.depth_confidence = matched.depth_confidence || track.depth_confidence;
        track.criticality = matched.criticality || track.criticality;
        track.risk = matched.risk !== undefined ? matched.risk : track.risk;
        track.priority = matched.priority || track.priority;
        track.distance_source = matched.distance_source || track.distance_source;
        track.frames_seen = (track.frames_seen || 1) + 1;
        track.lastSeen = now;
      }
    }

    // 2. Add new tracks for unmatched detections
    for (const idx of unmatchedIncoming) {
      const inc = incomingDetections[idx];
      const incDist = inc.estimated_distance !== undefined ? inc.estimated_distance : inc.distance_m;
      const incBox = inc.box || inc.bbox;

      this.tracks.push({
        id: this.nextTrackId++,
        classId: inc.classId,
        className: inc.className || inc.class || "Object",
        class: inc.className || inc.class || "Object",
        box: [...incBox],
        score: inc.score !== undefined ? inc.score : inc.confidence,
        confidence: inc.score !== undefined ? inc.score : inc.confidence,
        distance: incDist !== undefined ? incDist : null,
        distance_label: incDist !== null && incDist !== undefined ? `${Number(incDist).toFixed(1)} m` : (inc.distance_label || null),
        distance_source: inc.distance_source || "depth",
        distance_category: inc.distance_category || "FAR",
        direction: inc.direction || "center",
        horizontal_position: inc.horizontal_position || 0.0,
        corridor_overlap: inc.corridor_overlap || 0.0,
        path_relevance: inc.path_relevance || "outside path",
        depth_confidence: inc.depth_confidence || 0.5,
        criticality: inc.criticality || 0.5,
        risk: inc.risk || 0.0,
        priority: inc.priority || "LOW",
        frames_seen: 1,
        lastSeen: now
      });
    }

    // 3. Prune tracks that haven't been seen within trackTimeoutMs
    this.tracks = this.tracks.filter(t => (now - t.lastSeen) <= this.trackTimeoutMs);

    // 4. Return active stabilized detections
    return this.tracks.map(t => {
      const dist = t.distance !== null && !isNaN(t.distance) ? Math.round(t.distance * 10) / 10 : null;
      return {
        id: t.id,
        trackId: t.id,
        classId: t.classId,
        className: t.className,
        class: t.className,
        box: t.box.map(v => Math.round(v)),
        bbox: t.box.map(v => Math.round(v)),
        score: t.score,
        confidence: t.score,
        estimated_distance: dist,
        distance_m: dist,
        distance_label: dist !== null ? `${dist.toFixed(1)} m` : t.distance_label,
        distance_source: t.distance_source,
        distance_category: t.distance_category,
        direction: t.direction,
        horizontal_position: t.horizontal_position,
        corridor_overlap: t.corridor_overlap,
        path_relevance: t.path_relevance,
        depth_confidence: t.depth_confidence,
        criticality: t.criticality,
        risk: t.risk,
        priority: t.priority,
        frames_seen: t.frames_seen,
        isInterpolated: (now - t.lastSeen) > 120
      };
    });
  }

  reset() {
    this.tracks = [];
    this.nextTrackId = 1;
  }
}
