/**
 * YOLO Detection Post-Processor and Non-Maximum Suppression (NMS).
 * Supports standard YOLOv8/v11/v26 tensor layout:
 * Shape: [1, 4 + numClasses, numAnchors]
 */

export class YOLOVisualPostProcessor {
  constructor(classNames = []) {
    this.classNames = classNames;
  }

  setClasses(classNames) {
    this.classNames = classNames;
  }

  /**
   * Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]
   */
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

  /**
   * Non-Maximum Suppression (NMS)
   */
  applyNMS(candidates, iouThreshold = 0.45) {
    // Sort descending by score
    candidates.sort((a, b) => b.score - a.score);
    const selected = [];

    for (let i = 0; i < candidates.length; i++) {
      const current = candidates[i];
      let keep = true;

      for (let j = 0; j < selected.length; j++) {
        // Suppress if same class and high IoU
        if (selected[j].classId === current.classId) {
          const iou = this.computeIoU(selected[j].box, current.box);
          if (iou > iouThreshold) {
            keep = false;
            break;
          }
        }
      }

      if (keep) {
        selected.push(current);
      }
    }

    return selected;
  }

  /**
   * Decode ONNX model output tensor
   * @param {Float32Array} outputData - Raw tensor data
   * @param {number[]} outputShape - e.g. [1, 31, 4725]
   * @param {number} inputRes - Model resolution (e.g. 480)
   * @param {number} frameWidth - Live camera canvas width
   * @param {number} frameHeight - Live camera canvas height
   * @param {number} confThreshold - Minimum score cutoff (e.g. 0.25)
   * @param {number} iouThreshold - NMS IoU threshold (e.g. 0.45)
   */
  decode(outputData, outputShape, inputRes, frameWidth, frameHeight, confThreshold = 0.25, iouThreshold = 0.45) {
    const scaleX = frameWidth / inputRes;
    const scaleY = frameHeight / inputRes;
    const candidates = [];

    // FORMAT 1: End-to-End format [1, numBoxes, 6] (e.g. YOLOv10 / YOLO26 End2End)
    if (outputShape.length === 3 && outputShape[2] === 6) {
      const numBoxes = outputShape[1];
      for (let i = 0; i < numBoxes; i++) {
        const offset = i * 6;
        const score = outputData[offset + 4];
        if (score >= confThreshold) {
          const x1 = Math.max(0, outputData[offset + 0] * scaleX);
          const y1 = Math.max(0, outputData[offset + 1] * scaleY);
          const x2 = Math.min(frameWidth, outputData[offset + 2] * scaleX);
          const y2 = Math.min(frameHeight, outputData[offset + 3] * scaleY);
          if (x2 <= x1 || y2 <= y1) continue;
          const classId = Math.round(outputData[offset + 5]);

          candidates.push({
            box: [x1, y1, x2, y2],
            score: score,
            classId: classId,
            className: this.classNames[classId] || `Class ${classId}`
          });
        }
      }
      return candidates;
    }

    // FORMAT 2: Anchor layout [1, 4 + numClasses, anchors]
    const [, channels, anchors] = outputShape;
    const numClasses = channels - 4;

    for (let a = 0; a < anchors; a++) {
      let maxScore = -Infinity;
      let topClassId = -1;

      for (let c = 0; c < numClasses; c++) {
        const score = outputData[(4 + c) * anchors + a];
        if (score > maxScore) {
          maxScore = score;
          topClassId = c;
        }
      }

      if (maxScore >= confThreshold) {
        const cx = outputData[0 * anchors + a];
        const cy = outputData[1 * anchors + a];
        const w = outputData[2 * anchors + a];
        const h = outputData[3 * anchors + a];

        const x1 = Math.max(0, (cx - w / 2) * scaleX);
        const y1 = Math.max(0, (cy - h / 2) * scaleY);
        const x2 = Math.min(frameWidth, (cx + w / 2) * scaleX);
        const y2 = Math.min(frameHeight, (cy + h / 2) * scaleY);
        if (x2 <= x1 || y2 <= y1) continue;

        candidates.push({
          box: [x1, y1, x2, y2],
          score: maxScore,
          classId: topClassId,
          className: this.classNames[topClassId] || `Class ${topClassId}`
        });
      }
    }

    return this.applyNMS(candidates, iouThreshold);
  }
}
