'use client';

import { useCallback, useRef, useState } from 'react';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Detection {
  class: string;
  score: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
}

interface CocoModel {
  detect: (video: HTMLVideoElement, maxDetections?: number, minScore?: number) => Promise<Detection[]>;
}

type ObjectDistance = 'very close' | 'close' | 'near' | 'far' | 'uncertain';
type ObjectDirection = 'left' | 'center' | 'right';

interface DetectedObject {
  name: string;
  distance: ObjectDistance;
  direction: ObjectDirection;
  confidence: number;
}

interface ScanResult {
  scene_summary: string;
  objects: DetectedObject[];
  hazards: string[];
  action_advice: string;
}

interface FrameAnalysis {
  mean: number;
  variance: number;
  isBlank: boolean;
  isLowTexture: boolean;
}

const analyzeFrame = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
): FrameAnalysis | null => {
  if (width === 0 || height === 0) return null;
  try {
    const stride = Math.max(1, Math.floor(Math.min(width, height) / 36));
    const { data } = ctx.getImageData(0, 0, width, height);
    let count = 0;
    let mean = 0;
    let m2 = 0;

    for (let y = 0; y < height; y += stride) {
      for (let x = 0; x < width; x += stride) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        count += 1;
        const delta = lum - mean;
        mean += delta / count;
        m2 += delta * (lum - mean);
      }
    }

    if (count === 0) return null;
    const variance = m2 / count;
    const tooDark = mean < 10;
    const tooBright = mean > 245;
    const tooFlat = variance < 6;
    const isBlank = (tooDark || tooBright) && tooFlat;
    const isLowTexture = variance < 18 && mean > 12 && mean < 240;
    return { mean, variance, isBlank, isLowTexture };
  } catch {
    return null;
  }
};

const buildBlockedResult = (): ScanResult => ({
  scene_summary: 'Camera view appears blocked or too dark.',
  objects: [],
  hazards: ['camera view blocked or too dark'],
  action_advice: 'Clear the camera view and scan again.',
});

const buildLowTextureResult = (): ScanResult => ({
  scene_summary: 'Limited detail detected ahead.',
  objects: [],
  hazards: ['possible wall or obstruction ahead'],
  action_advice: 'Proceed slowly and rescan.',
});

// ─── Object classification ───────────────────────────────────────────────────

// Objects that represent immediate danger for a blind person
const HIGH_DANGER_OBJECTS = new Set([
  'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train',
  'fire hydrant', 'stop sign', 'traffic light',
  'dog', 'horse', 'cow', 'elephant', 'bear',
  'knife', 'scissors',
]);

// Objects that require caution
const MEDIUM_CAUTION_OBJECTS = new Set([
  'person', 'cat', 'bird',
  'chair', 'couch', 'bench', 'dining table', 'bed',
  'potted plant', 'backpack', 'suitcase', 'umbrella',
  'skateboard', 'surfboard', 'snowboard', 'skis',
  'bottle', 'cup', 'bowl',
  'toilet', 'sink',
  'oven', 'microwave', 'refrigerator', 'toaster',
]);


const getDirection = (
  bboxCenterX: number,
  frameWidth: number
): ObjectDirection => {
  const relativeX = bboxCenterX / frameWidth;
  if (relativeX < 0.33) return 'left';
  if (relativeX > 0.67) return 'right';
  return 'center';
};

const getDistanceCategory = (
  bboxHeight: number,
  frameHeight: number
): ObjectDistance => {
  if (frameHeight <= 0) return 'uncertain';
  const ratio = bboxHeight / frameHeight;
  if (ratio > 0.7) return 'very close';
  if (ratio > 0.45) return 'close';
  if (ratio > 0.2) return 'near';
  return 'far';
};

const directionForHazard = (direction: ObjectDirection): 'left' | 'right' | 'ahead' => {
  if (direction === 'center') return 'ahead';
  return direction;
};

const isHazardObject = (objectClass: string, distance: ObjectDistance): boolean => {
  if (HIGH_DANGER_OBJECTS.has(objectClass)) return distance !== 'far';
  if (MEDIUM_CAUTION_OBJECTS.has(objectClass)) {
    return distance === 'very close' || distance === 'close';
  }
  return false;
};

const buildSceneSummary = (objects: DetectedObject[]): string => {
  if (objects.length === 0) return 'No major objects detected.';
  const top = objects.slice(0, 3)
    .map((obj) => `${obj.name} ${obj.distance} ${obj.direction}`)
    .join(', ');
  return `Detected ${top}.`;
};

const buildActionAdvice = (hazards: string[], objects: DetectedObject[]): string => {
  if (hazards.length > 0) {
    return `Caution: ${hazards[0]}.`;
  }
  if (objects.length > 0) {
    const lead = objects[0];
    return `Proceed carefully. ${lead.name} ${lead.distance} ${lead.direction}.`;
  }
  return 'No clear obstacles detected. Proceed cautiously.';
};

// ─── Build scan result from detections ────────────────────────────────────

export const buildScanResultFromDetections = (
  detections: Detection[],
  frameWidth: number,
  frameHeight: number,
  frameAnalysis?: FrameAnalysis | null
): ScanResult => {
  if (frameAnalysis?.isBlank) {
    return buildBlockedResult();
  }
  if (detections.length === 0) {
    if (frameAnalysis?.isLowTexture) {
      return buildLowTextureResult();
    }
    return {
      scene_summary: 'No major objects detected.',
      objects: [],
      hazards: [],
      action_advice: 'No clear obstacles detected. Proceed cautiously.',
    };
  }

  const frameArea = frameWidth * frameHeight;

  const objects = detections
    .map((det) => {
      const [x, y, w, h] = det.bbox;
      const bboxSize = (w * h) / frameArea;
      const centerX = x + w / 2;
      const direction = getDirection(centerX, frameWidth);
      const distance = getDistanceCategory(h, frameHeight);
      return {
        name: det.class,
        distance,
        direction,
        confidence: Math.max(0, Math.min(1, det.score)),
        _bboxSize: bboxSize,
      };
    })
    .sort((a, b) => b._bboxSize - a._bboxSize)
    .slice(0, 10)
    .map(({ name, distance, direction, confidence }) => ({
      name,
      distance,
      direction,
      confidence,
    }));

  const hazards = objects
    .filter((obj) => isHazardObject(obj.name, obj.distance))
    .map((obj) => `${obj.name} ${obj.distance} ${directionForHazard(obj.direction)}`)
    .slice(0, 6);

  const scene_summary = buildSceneSummary(objects);
  const action_advice = buildActionAdvice(hazards, objects);

  return {
    scene_summary,
    objects,
    hazards,
    action_advice,
  };
};

// ─── Hook ────────────────────────────────────────────────────────────────────

export function useObjectDetection() {
  const modelRef = useRef<CocoModel | null>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);

  const loadModel = useCallback(async () => {
    if (modelRef.current || modelLoading) return;
    setModelLoading(true);
    setModelError(null);

    try {
      const offlineOnly = (process.env.NEXT_PUBLIC_OFFLINE_ONLY ?? 'true').toLowerCase() !== 'false';
      const configuredModelUrl = process.env.NEXT_PUBLIC_COCO_SSD_MODEL_URL;
      const localModelUrl = configuredModelUrl || '/models/coco-ssd/model.json';

      // Dynamic imports to avoid SSR issues
      const tf = await import('@tensorflow/tfjs');
      
      // Set backend - prefer WebGL for performance
      if (tf.getBackend() !== 'webgl') {
        try {
          await tf.setBackend('webgl');
          await tf.ready();
        } catch {
          // Fallback to WASM or CPU
          try {
            await tf.setBackend('wasm');
            await tf.ready();
          } catch {
            await tf.setBackend('cpu');
            await tf.ready();
          }
        }
      }

      const cocoSsd = await import('@tensorflow-models/coco-ssd');
      const model = await cocoSsd.load(
        offlineOnly || configuredModelUrl
          ? { base: 'lite_mobilenet_v2', modelUrl: localModelUrl }
          : { base: 'lite_mobilenet_v2' }
      );

      modelRef.current = model as unknown as CocoModel;
      setModelLoaded(true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load detection model';
      const hint = (process.env.NEXT_PUBLIC_OFFLINE_ONLY ?? 'true').toLowerCase() !== 'false'
        ? 'Local model not found. Run npm run download:model and set NEXT_PUBLIC_COCO_SSD_MODEL_URL.'
        : '';
      setModelError(hint ? `${msg}. ${hint}` : msg);
      console.error('COCO-SSD load error:', err);
    } finally {
      setModelLoading(false);
    }
  }, [modelLoading]);

  const detect = useCallback(
    async (video: HTMLVideoElement): Promise<ScanResult | null> => {
      const model = modelRef.current;
      if (!model) return null;
      if (video.videoWidth === 0 || video.videoHeight === 0) return null;

      let frameAnalysis: FrameAnalysis | null = null;
      try {
        const canvas = analysisCanvasRef.current ?? document.createElement('canvas');
        analysisCanvasRef.current = canvas;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const sampleWidth = Math.min(320, video.videoWidth);
          const scale = sampleWidth / video.videoWidth;
          const sampleHeight = Math.max(1, Math.round(video.videoHeight * scale));
          canvas.width = sampleWidth;
          canvas.height = sampleHeight;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          frameAnalysis = analyzeFrame(ctx, canvas.width, canvas.height);
        }
      } catch {
        frameAnalysis = null;
      }

      if (frameAnalysis?.isBlank) {
        return buildBlockedResult();
      }

      try {
        const detections = await model.detect(video, 20, 0.35);
        return buildScanResultFromDetections(
          detections as Detection[],
          video.videoWidth,
          video.videoHeight,
          frameAnalysis
        );
      } catch (err) {
        console.error('Detection error:', err);
        return null;
      }
    },
    []
  );

  return {
    loadModel,
    detect,
    modelLoaded,
    modelLoading,
    modelError,
  };
}
