/**
 * Post-Processing for YOLOv8/v11/v26 ONNX Models.
 * Handles transposed and standard output shapes:
 * - [1, 4 + numClasses, N] (standard YOLOv8 transposed output)
 * - [1, N, 4 + numClasses] (pre-transposed)
 * Decodes bounding boxes, filters by confidence, and applies Non-Maximum Suppression (NMS).
 */

import { distanceEstimator } from "./distance.js";

export class YOLOVisualPostProcessor {
  constructor(classNames = []) {
    this.classNames = classNames;
    this.iouThreshold = 0.45;
  }

  setClasses(classNames) {
    this.classNames = classNames;
  }

  decode(outputData, dims, modelRes, frameWidth, frameHeight, confThreshold = 0.25) {
    const detections = [];
    if (!outputData || !dims || dims.length < 2) return detections;

    let numChannels, numAnchors, isTransposed;

    if (dims.length === 3) {
      // e.g. [1, 31, 8400] or [1, 8400, 31]
      const [, d1, d2] = dims;
      if (d1 < d2) {
        numChannels = d1;
        numAnchors = d2;
        isTransposed = true; // [1, channels, anchors]
      } else {
        numChannels = d2;
        numAnchors = d1;
        isTransposed = false; // [1, anchors, channels]
      }
    } else {
      return detections;
    }

    const numClasses = numChannels - 4;
    const scaleX = frameWidth / modelRes;
    const scaleY = frameHeight / modelRes;

    const candidateBoxes = [];
    const candidateScores = [];
    const candidateClasses = [];

    for (let a = 0; a < numAnchors; a++) {
      let maxScore = -Infinity;
      let maxClass = -1;

      for (let c = 0; c < numClasses; c++) {
        const scoreIdx = isTransposed
          ? (4 + c) * numAnchors + a
          : a * numChannels + (4 + c);
        const score = outputData[scoreIdx];
        if (score > maxScore) {
          maxScore = score;
          maxClass = c;
        }
      }

      if (maxScore >= confThreshold) {
        let cx, cy, w, h;
        if (isTransposed) {
          cx = outputData[0 * numAnchors + a];
          cy = outputData[1 * numAnchors + a];
          w = outputData[2 * numAnchors + a];
          h = outputData[3 * numAnchors + a];
        } else {
          cx = outputData[a * numChannels + 0];
          cy = outputData[a * numChannels + 1];
          w = outputData[a * numChannels + 2];
          h = outputData[a * numChannels + 3];
        }

        const x1 = Math.max(0, (cx - w / 2) * scaleX);
        const y1 = Math.max(0, (cy - h / 2) * scaleY);
        const x2 = Math.min(frameWidth, (cx + w / 2) * scaleX);
        const y2 = Math.min(frameHeight, (cy + h / 2) * scaleY);

        candidateBoxes.push([x1, y1, x2, y2]);
        candidateScores.push(maxScore);
        candidateClasses.push(maxClass);
      }
    }

    // Apply Non-Maximum Suppression (NMS)
    const keepIndices = this.nms(candidateBoxes, candidateScores, this.iouThreshold);

    for (const idx of keepIndices) {
      const box = candidateBoxes[idx];
      const score = candidateScores[idx];
      const clsId = candidateClasses[idx];
      const className = this.classNames[clsId] || `Class ${clsId}`;

      const est = distanceEstimator.estimate(box, frameWidth, frameHeight);

      detections.push({
        box,
        score,
        classId: clsId,
        className,
        distanceLabel: est.label,
        distanceCategory: est.category,
        distanceM: est.distanceM,
        areaRatio: est.areaRatio,
        priority: est.priority
      });
    }

    return detections;
  }

  nms(boxes, scores, iouThresh) {
    const indices = Array.from({ length: scores.length }, (_, i) => i);
    indices.sort((a, b) => scores[b] - scores[a]);

    const keep = [];
    const suppressed = new Uint8Array(boxes.length);

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (suppressed[idx]) continue;
      keep.push(idx);

      for (let j = i + 1; j < indices.length; j++) {
        const nextIdx = indices[j];
        if (suppressed[nextIdx]) continue;

        const iou = this.calculateIoU(boxes[idx], boxes[nextIdx]);
        if (iou >= iouThresh) {
          suppressed[nextIdx] = 1;
        }
      }
    }

    return keep;
  }

  calculateIoU(b1, b2) {
    const [x1A, y1A, x2A, y2A] = b1;
    const [x1B, y1B, x2B, y2B] = b2;

    const interX1 = Math.max(x1A, x1B);
    const interY1 = Math.max(y1A, y1B);
    const interX2 = Math.min(x2A, x2B);
    const interY2 = Math.min(y2A, y2B);

    const interW = Math.max(0, interX2 - interX1);
    const interH = Math.max(0, interY2 - interY1);
    const interArea = interW * interH;

    const areaA = (x2A - x1A) * (y2A - y1A);
    const areaB = (x2B - x1B) * (y2B - y1B);
    const unionArea = areaA + areaB - interArea;

    return unionArea > 0 ? interArea / unionArea : 0;
  }
}
