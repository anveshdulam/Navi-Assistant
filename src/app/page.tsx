'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useTheme } from '@/components/theme-provider';
import {
  Camera,
  CameraOff,
  Eye,
  AlertTriangle,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Volume2,
  Play,
  Pause,
  RotateCcw,
  Moon,
  Sun,
  Zap,
  RefreshCw,
  Activity,
  Scan,
  Radio,
  Navigation,
  Mic,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import ARIAStatus from '@/components/ARIA-Status';
import { useObjectDetection } from '@/hooks/useObjectDetection';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}

type ObjectDistance = 'very close' | 'close' | 'near' | 'far' | 'uncertain';
type ObjectDirection = 'left' | 'center' | 'right' | 'ahead';
type AlertLevel = 'clear' | 'caution' | 'danger';

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

interface ImageCaptureLike {
  grabFrame: () => Promise<ImageBitmap>;
}

type ImageCaptureConstructor = new (track: MediaStreamTrack) => ImageCaptureLike;

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((event: any) => void) | null;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: any) => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

const SCAN_FREQUENCIES = [4000, 6000, 10000] as const;
type ScanFrequency = (typeof SCAN_FREQUENCIES)[number];

const DEFAULT_SCAN_FREQUENCY: ScanFrequency = 4000;
const MAX_SCAN_RETRIES = 2;
const CAMERA_STORAGE_KEY = 'navassist-camera-device';
const RATE_LIMIT_COOLDOWN_MS = 12000;
const LOCAL_VISION_STORAGE_KEY = 'navassist-local-vision';
const VOICE_COMMAND_DEBOUNCE_MS = 1500;

const normalizeFrequency = (value?: number): ScanFrequency => {
  if (SCAN_FREQUENCIES.includes(value as ScanFrequency)) {
    return value as ScanFrequency;
  }
  return DEFAULT_SCAN_FREQUENCY;
};

const isFrameBlank = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
) => {
  if (width === 0 || height === 0) return true;
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

    if (count === 0) return true;
    const variance = m2 / count;
    const tooDark = mean < 8;
    const tooBright = mean > 245;
    const tooFlat = variance < 6;
    return (tooDark || tooBright) && tooFlat;
  } catch {
    return false;
  }
};

const buildSpokenAlert = (result: ScanResult) => {
  const baseAlert = typeof result.action_advice === 'string'
    ? result.action_advice.trim()
    : '';
  if (baseAlert) return baseAlert;
  const summary = typeof result.scene_summary === 'string'
    ? result.scene_summary.trim()
    : '';
  return summary || 'Path appears clear ahead.';
};

const getAlertLevel = (result: ScanResult | null): AlertLevel => {
  if (!result || !Array.isArray(result.hazards) || result.hazards.length === 0) {
    return 'clear';
  }
  const hazardText = result.hazards.join(' ').toLowerCase();
  if (hazardText.includes('very close') || hazardText.includes('close')) {
    return 'danger';
  }
  return 'caution';
};

const objectUrgency = (distance: ObjectDistance): 'high' | 'medium' | 'low' => {
  if (distance === 'very close' || distance === 'close') return 'high';
  if (distance === 'near') return 'medium';
  return 'low';
};

const isImmediateDanger = (result: ScanResult) => {
  const level = getAlertLevel(result);
  if (level !== 'danger') return false;
  return result.hazards.some((hazard) => hazard.toLowerCase().includes('ahead'));
};

// ─── Component ───────────────────────────────────────────────────────────────

export default function BlindNavigationAssistant() {
  // Theme
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Local object detection (TensorFlow.js COCO-SSD)
  const { loadModel, detect, modelLoaded, modelLoading, modelError } = useObjectDetection();

  // Camera
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Scanning
  const [isScanning, setIsScanning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [scanFrequency, setScanFrequency] = useState<ScanFrequency>(
    DEFAULT_SCAN_FREQUENCY
  );
  const [scanCount, setScanCount] = useState(0);
  const [isManualScan, setIsManualScan] = useState(false);
  const scanTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isScanningRef = useRef(false);
  const rateLimitedUntilRef = useRef(0);

  // Results
  const [currentResult, setCurrentResult] = useState<ScanResult | null>(null);
  const [scanHistory, setScanHistory] = useState<ScanResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);

  // Animation
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Accessibility + preferences
  const [ariaStatus, setAriaStatus] = useState('');
  const [hapticEnabled, setHapticEnabled] = useState(true);
  const [localVisionEnabled, setLocalVisionEnabled] = useState(false);
  const [localVisionPrefLoaded, setLocalVisionPrefLoaded] = useState(false);
  const [voiceCommandsEnabled, setVoiceCommandsEnabled] = useState(true);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const [voiceListening, setVoiceListening] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [voiceNeedsGesture, setVoiceNeedsGesture] = useState(false);
  const [userId, setUserId] = useState<string | null>(null);
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [cameraReady, setCameraReady] = useState(false);
  const [healthStatus, setHealthStatus] = useState<string | null>(null);
  const [healthChecking, setHealthChecking] = useState(false);
  const [detectionMode, setDetectionMode] = useState<'local' | 'api'>('local');

  const cameraActiveRef = useRef(cameraActive);
  const isPausedRef = useRef(isPaused);
  const isManualScanRef = useRef(isManualScan);
  const hapticEnabledRef = useRef(hapticEnabled);
  const userIdRef = useRef<string | null>(userId);
  const cameraReadyRef = useRef(cameraReady);
  const localVisionEnabledRef = useRef(localVisionEnabled);
  const voiceCommandsEnabledRef = useRef(voiceCommandsEnabled);
  const isSpeakingRef = useRef(isSpeaking);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const voiceRestartTimerRef = useRef<number | null>(null);
  const lastVoiceCommandRef = useRef({ text: '', time: 0 });

  const startButtonRef = useRef<HTMLButtonElement>(null);
  const scanButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => { cameraActiveRef.current = cameraActive; }, [cameraActive]);
  useEffect(() => { isPausedRef.current = isPaused; }, [isPaused]);
  useEffect(() => { isManualScanRef.current = isManualScan; }, [isManualScan]);
  useEffect(() => { hapticEnabledRef.current = hapticEnabled; }, [hapticEnabled]);
  useEffect(() => { userIdRef.current = userId; }, [userId]);
  useEffect(() => { cameraReadyRef.current = cameraReady; }, [cameraReady]);
  useEffect(() => { localVisionEnabledRef.current = localVisionEnabled; }, [localVisionEnabled]);
  useEffect(() => { voiceCommandsEnabledRef.current = voiceCommandsEnabled; }, [voiceCommandsEnabled]);
  useEffect(() => { isSpeakingRef.current = isSpeaking; }, [isSpeaking]);

  const announceStatus = useCallback((message: string) => {
    if (!message) return;
    setAriaStatus((prev) => (prev === message ? `${message} ` : message));
  }, []);

  const getSpeechRecognitionCtor = useCallback(() => {
    if (typeof window === 'undefined') return null;
    const w = window as Window & {
      SpeechRecognition?: SpeechRecognitionConstructor;
      webkitSpeechRecognition?: SpeechRecognitionConstructor;
    };
    return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
  }, []);

  useEffect(() => {
    if (!mounted) return;
    setVoiceSupported(Boolean(getSpeechRecognitionCtor()));
  }, [getSpeechRecognitionCtor, mounted]);

  useEffect(() => {
    if (!mounted) return;
    const stored = window.localStorage.getItem(LOCAL_VISION_STORAGE_KEY);
    if (stored !== null) {
      setLocalVisionEnabled(stored === 'true');
    }
    setLocalVisionPrefLoaded(true);
  }, [mounted]);

  useEffect(() => {
    if (!mounted || !localVisionPrefLoaded) return;
    window.localStorage.setItem(LOCAL_VISION_STORAGE_KEY, String(localVisionEnabled));
  }, [localVisionEnabled, localVisionPrefLoaded, mounted]);

  useEffect(() => {
    if (!mounted) return;
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch(() => { return; });
    }
  }, [mounted]);

  useEffect(() => {
    if (!mounted) return;
    const stored = window.localStorage.getItem('navassist-user-id');
    if (stored) { setUserId(stored); return; }
    const fallbackId = `navassist-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const newId =
      typeof crypto !== 'undefined' && 'randomUUID' in crypto
        ? crypto.randomUUID()
        : fallbackId;
    window.localStorage.setItem('navassist-user-id', newId);
    setUserId(newId);
  }, [mounted]);

  const loadPreferences = useCallback(async (id: string) => {
    try {
      const response = await fetch('/api/vision-scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'get-preferences', userId: id }),
      });
      const payload = (await response.json()) as ApiResponse<{
        scanFrequency: number;
        hapticEnabled: boolean;
      }>;
      if (response.ok && payload.success && payload.data) {
        setScanFrequency(normalizeFrequency(payload.data.scanFrequency));
        setHapticEnabled(Boolean(payload.data.hapticEnabled));
      }
    } catch { return; } finally { setPreferencesLoaded(true); }
  }, []);

  const savePreferences = useCallback(
    async (id: string, frequency: ScanFrequency, haptic: boolean) => {
      try {
        await fetch('/api/vision-scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'update-preferences', userId: id,
            scanFrequency: frequency, hapticEnabled: haptic,
          }),
        });
      } catch { return; }
    }, []
  );

  useEffect(() => { if (!userId) return; loadPreferences(userId); }, [loadPreferences, userId]);
  useEffect(() => {
    if (!userId || !preferencesLoaded) return;
    savePreferences(userId, scanFrequency, hapticEnabled);
  }, [hapticEnabled, preferencesLoaded, savePreferences, scanFrequency, userId]);

  useEffect(() => {
    if (!mounted) return;
    if (cameraError) { announceStatus(cameraError); return; }
    if (cameraActive) { announceStatus('Camera started. Auto scan ready.'); }
    else { announceStatus('Camera stopped.'); }
  }, [announceStatus, cameraActive, cameraError, mounted]);

  useEffect(() => {
    if (!mounted || !cameraActive) return;
    if (isPaused) { announceStatus('Scanning paused.'); }
    else { announceStatus('Scanning resumed.'); }
  }, [announceStatus, cameraActive, isPaused, mounted]);

  useEffect(() => {
    if (!mounted) return;
    if (isScanning) { announceStatus('Scanning in progress.'); }
  }, [announceStatus, isScanning, mounted]);

  useEffect(() => { if (error) { announceStatus(error); } }, [announceStatus, error]);
  useEffect(() => { if (currentResult) { announceStatus(buildSpokenAlert(currentResult)); } }, [announceStatus, currentResult]);
  useEffect(() => { if (healthStatus) { announceStatus(healthStatus); } }, [announceStatus, healthStatus]);

  useEffect(() => {
    if (!mounted) return;
    if (cameraActive) { scanButtonRef.current?.focus(); }
    else { startButtonRef.current?.focus(); }
  }, [cameraActive, mounted]);

  useEffect(() => {
    if (!mounted) return;
    const storedId = window.localStorage.getItem(CAMERA_STORAGE_KEY) ?? '';
    navigator.mediaDevices.enumerateDevices()
      .then((devices) => {
        const inputs = devices.filter((device) => device.kind === 'videoinput');
        setCameraDevices(inputs);
        const selectedMatches = selectedDeviceId && inputs.some((device) => device.deviceId === selectedDeviceId);
        if (selectedMatches) return;
        const storedMatches = storedId && inputs.some((device) => device.deviceId === storedId);
        if (storedMatches) { setSelectedDeviceId(storedId); return; }
        if (inputs[0]) { setSelectedDeviceId(inputs[0].deviceId); }
      })
      .catch(() => { setCameraDevices([]); });
  }, [mounted, selectedDeviceId]);

  // ─── Theme toggle ───────────────────────────────────────────────────────
  const toggleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  }, [theme, setTheme]);

  // ─── Camera ─────────────────────────────────────────────────────────────
  const waitForVideoReady = useCallback((timeoutMs = 3000) => {
    return new Promise<boolean>((resolve) => {
      const video = videoRef.current;
      if (!video) { resolve(false); return; }
      const hasFrames = () => video.videoWidth > 0 && video.videoHeight > 0;
      if (video.readyState >= 2 && hasFrames()) { resolve(true); return; }
      const timeoutId = window.setTimeout(() => { cleanup(); resolve(false); }, timeoutMs);
      const handleReady = () => { if (!hasFrames()) return; cleanup(); resolve(true); };
      const handleError = () => { cleanup(); resolve(false); };
      const cleanup = () => {
        clearTimeout(timeoutId);
        video.removeEventListener('loadedmetadata', handleReady);
        video.removeEventListener('canplay', handleReady);
        video.removeEventListener('loadeddata', handleReady);
        video.removeEventListener('playing', handleReady);
        video.removeEventListener('timeupdate', handleReady);
        video.removeEventListener('error', handleError);
      };
      const requestFrame =
        'requestVideoFrameCallback' in video
          ? (cb: () => void) =>
              (video as HTMLVideoElement & {
                requestVideoFrameCallback?: (callback: () => void) => void;
              }).requestVideoFrameCallback?.(cb)
          : null;
      if (requestFrame) {
        requestFrame(() => { if (hasFrames()) { cleanup(); resolve(true); } });
      }
      video.addEventListener('loadedmetadata', handleReady, { once: true });
      video.addEventListener('canplay', handleReady, { once: true });
      video.addEventListener('loadeddata', handleReady, { once: true });
      video.addEventListener('playing', handleReady, { once: true });
      video.addEventListener('timeupdate', handleReady, { once: true });
      video.addEventListener('error', handleError, { once: true });
    });
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current.load();
    }
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((device) => device.kind === 'videoinput');
      const video = videoRef.current;
      if (!video) { throw new Error('Camera element not ready.'); }

      const attachStream = async (stream: MediaStream) => {
        streamRef.current = stream;
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();
        await new Promise((resolve) => requestAnimationFrame(resolve));
        const ready = await waitForVideoReady(3500);
        if (!ready) { stream.getTracks().forEach((track) => track.stop()); return false; }
        return true;
      };

      const tryConstraints = async (constraints: MediaStreamConstraints) => {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        const ready = await attachStream(stream);
        if (!ready) { stream.getTracks().forEach((track) => track.stop()); }
        return ready;
      };

      const constraintsList: MediaStreamConstraints[] = [
        ...(selectedDeviceId
          ? [{ video: { deviceId: { exact: selectedDeviceId } }, audio: false } as MediaStreamConstraints]
          : []),
        {
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        },
        ...videoInputs.map((device) => ({ video: { deviceId: { exact: device.deviceId } }, audio: false })),
        { video: true, audio: false },
      ];

      for (const constraints of constraintsList) {
        try {
          const ready = await tryConstraints(constraints);
          if (ready) {
            // BUG FIX: was setCameraReady(false) — must be true after successful start
            setCameraReady(true);
            setCameraActive(true);
            return true;
          }
        } catch { continue; }
      }

      throw new Error('Camera stream is not delivering frames.');
    } catch (err) {
      const message =
        err instanceof DOMException
          ? err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError'
            ? 'Camera access denied. Please grant camera permission in your browser settings and reload the page.'
            : `Camera error: ${err.message}`
          : err instanceof Error
            ? err.message
            : 'Failed to access camera. Please ensure your device has a camera and try again.';
      setCameraError(message);
      setCameraActive(false);
      return false;
    }
  }, [selectedDeviceId, waitForVideoReady]);

  const stopAutoScan = useCallback(() => {
    if (scanTimerRef.current) { clearInterval(scanTimerRef.current); scanTimerRef.current = null; }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) { streamRef.current.getTracks().forEach((track) => track.stop()); streamRef.current = null; }
    if (videoRef.current) { videoRef.current.pause(); videoRef.current.srcObject = null; videoRef.current.load(); }
    isScanningRef.current = false;
    setCameraActive(false);
    setCameraReady(false);
    setIsScanning(false);
    setIsPaused(false);
    setRetrying(false);
    stopAutoScan();
  }, [stopAutoScan]);

  // ─── Initialize ─────────────────────────────────────────────────────────
  useEffect(() => {
    setMounted(true);
    return () => {
      stopCamera();
      stopAutoScan();
      if ('speechSynthesis' in window) { window.speechSynthesis.cancel(); }
    };
  }, [stopAutoScan, stopCamera]);

  // ─── Capture frame (only needed for API mode) ──────────────────────────
  const captureFrame = useCallback(async (): Promise<string | null> => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const waitForNextFrame = () =>
      new Promise<void>((resolve) => {
        const video = videoRef.current;
        if (!video) { resolve(); return; }
        const requestFrame =
          'requestVideoFrameCallback' in video
            ? (cb: () => void) =>
                (video as HTMLVideoElement & {
                  requestVideoFrameCallback?: (callback: () => void) => void;
                }).requestVideoFrameCallback?.(cb)
            : null;
        if (requestFrame) { requestFrame(() => resolve()); return; }
        requestAnimationFrame(() => { window.setTimeout(() => resolve(), 40); });
      });

    const ImageCaptureCtor = (
      window as Window & { ImageCapture?: ImageCaptureConstructor }
    ).ImageCapture;

    if (ImageCaptureCtor && streamRef.current) {
      const track = streamRef.current.getVideoTracks()[0];
      if (track) {
        for (let attempt = 0; attempt < 2; attempt += 1) {
          try {
            const capture = new ImageCaptureCtor(track);
            const bitmap = await capture.grabFrame();
            const ctx = canvas.getContext('2d');
            if (!ctx) return null;
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
            ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            const blank = isFrameBlank(ctx, canvas.width, canvas.height);
            if ('close' in bitmap) { bitmap.close(); }
            if (blank) { await waitForNextFrame(); continue; }
            return canvas.toDataURL('image/jpeg', 0.92);
          } catch { await waitForNextFrame(); }
        }
      }
    }

    const video = videoRef.current;
    if (!video) return null;

    if (video.readyState < 2) { await waitForNextFrame(); }

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      const ready = await waitForVideoReady(2500);
      if (!ready) {
        for (let attempt = 0; attempt < 3; attempt += 1) { await waitForNextFrame(); }
      }
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        const track = streamRef.current?.getVideoTracks()[0];
        const settings = track?.getSettings();
        if (settings?.width && settings?.height) {
          canvas.width = settings.width;
          canvas.height = settings.height;
        } else {
          const bounds = video.getBoundingClientRect();
          if (bounds.width > 0 && bounds.height > 0) {
            canvas.width = Math.round(bounds.width);
            canvas.height = Math.round(bounds.height);
          } else { return null; }
        }
      }
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Try createImageBitmap for a more reliable snapshot
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const bitmap = await createImageBitmap(video);
        // Higher resolution for better recognition
        const maxWidth = 1280;
        const sourceWidth = bitmap.width || canvas.width || video.videoWidth;
        const sourceHeight = bitmap.height || canvas.height || video.videoHeight;
        const scale = Math.min(1, maxWidth / sourceWidth);
        canvas.width = sourceWidth * scale;
        canvas.height = sourceHeight * scale;
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        const blank = isFrameBlank(ctx, canvas.width, canvas.height);
        if ('close' in bitmap) { bitmap.close(); }
        if (blank) { await waitForNextFrame(); continue; }
        // Higher JPEG quality for better AI recognition
        return canvas.toDataURL('image/jpeg', 0.92);
      } catch { await waitForNextFrame(); }
    }

    // Fallback: draw directly from video
    const maxWidth = 1280;
    const sourceWidth = canvas.width || video.videoWidth;
    const sourceHeight = canvas.height || video.videoHeight;
    const scale = Math.min(1, maxWidth / sourceWidth);
    canvas.width = sourceWidth * scale;
    canvas.height = sourceHeight * scale;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    if (isFrameBlank(ctx, canvas.width, canvas.height)) {
      return null;
    }
    return canvas.toDataURL('image/jpeg', 0.92);
  }, [waitForVideoReady]);

  const warmupCamera = useCallback(async () => {
    setCameraReady(false);
    for (let attempt = 0; attempt < 6; attempt += 1) {
      const frame = await captureFrame();
      if (frame) { setCameraReady(true); return true; }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    setCameraError('Camera started but frames are not available. Try switching camera source.');
    return false;
  }, [captureFrame]);

  // ─── TTS ────────────────────────────────────────────────────────────────
  const speakAlert = useCallback(
    (text: string, threatLevel: AlertLevel) => {
      if (!('speechSynthesis' in window)) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      switch (threatLevel) {
        case 'danger':
          utterance.rate = 1.3; utterance.pitch = 1.4; utterance.volume = 1.0; break;
        case 'caution':
          utterance.rate = 1.1; utterance.pitch = 1.1; utterance.volume = 0.9; break;
        default:
          utterance.rate = 0.95; utterance.pitch = 1.0; utterance.volume = 0.8; break;
      }
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    }, []
  );

  const testVoice = useCallback(() => {
    speakAlert('Voice test successful. Navigation assistant is ready.', 'clear');
  }, [speakAlert]);

  const handleHealthCheck = useCallback(async () => {
    setHealthChecking(true);
    setHealthStatus('Checking vision model...');
    try {
      const response = await fetch('/api/vision-scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'health' }),
      });
      const payload = (await response.json()) as ApiResponse<{
        configured: boolean;
        provider: string | null;
        model?: string;
      }>;
      if (!response.ok || !payload?.success) {
        setHealthStatus(payload?.message || 'Health check failed.');
        return;
      }
      if (payload.data?.configured) {
        const provider = payload.data.provider ?? 'vision';
        const model = payload.data.model ? ` (${payload.data.model})` : '';
        setHealthStatus(`✓ Vision model ready: ${provider}${model}`);
      } else {
        setHealthStatus('✗ Vision model not configured.');
      }
    } catch { setHealthStatus('Health check failed — network error.'); }
    finally { setHealthChecking(false); }
  }, []);

  const triggerHaptics = useCallback((result: ScanResult) => {
    if (!hapticEnabledRef.current) return;
    if (!('vibrate' in navigator)) return;
    if (isImmediateDanger(result)) { navigator.vibrate([200, 120, 200]); }
    else { navigator.vibrate([50]); }
  }, []);

  const applyScanResult = useCallback(
    (result: ScanResult, mode: 'local' | 'api') => {
      const normalizedAlert = buildSpokenAlert(result);
      const normalizedResult: ScanResult = {
        ...result,
        action_advice: normalizedAlert,
        objects: Array.isArray(result.objects) ? result.objects : [],
        hazards: Array.isArray(result.hazards) ? result.hazards : [],
      };
      const alertLevel = getAlertLevel(normalizedResult);
      setDetectionMode(mode);
      setCurrentResult(normalizedResult);
      setScanHistory((prev) => [normalizedResult, ...prev].slice(0, 10));
      setScanCount((count) => count + 1);
      setError(null);
      setRetrying(false);
      triggerHaptics(normalizedResult);
      speakAlert(normalizedResult.action_advice, alertLevel);
    },
    [speakAlert, triggerHaptics]
  );

  // ─── Local detection (fast fallback - uses TensorFlow.js COCO-SSD) ─────
  const performLocalScan = useCallback(async (): Promise<ScanResult | null> => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return null;
    return detect(video);
  }, [detect]);

  // ─── Vision scan (offline local-first) ───────────────────────────────────
  const performScan = useCallback(async () => {
    if (isScanningRef.current) return false;
    isScanningRef.current = true;
    setIsScanning(true);
    setError(null);
    setRetrying(false);

    const finalizeScan = () => {
      isScanningRef.current = false;
      setIsScanning(false);
      setRetrying(false);
    };

    const attemptScan = async () => {
      if (!cameraReadyRef.current) {
        const warmed = await warmupCamera();
        if (!warmed) { throw new Error('Camera not ready. Please try again.'); }
      }
      if (!streamRef.current || !streamRef.current.active) {
        const restarted = await startCamera();
        if (!restarted) { throw new Error('Camera not ready. Please try again.'); }
        await new Promise((resolve) => setTimeout(resolve, 200));
      }
      let frame = await captureFrame();
      if (!frame) {
        const restarted = await startCamera();
        if (restarted) { frame = await captureFrame(); }
      }
      if (!frame) {
        throw new Error('Unable to capture frame. Ensure no other app is using the camera.');
      }

      let response: Response;
      try {
        response = await fetch('/api/vision-scan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: frame,
            userId: userIdRef.current,
            scanFrequency,
            hapticEnabled: hapticEnabledRef.current,
          }),
        });
      } catch {
        throw new Error('Scan failed. Check the local vision server and try again.');
      }

      // Handle rate limiting — don't retry, just wait
      if (response.status === 429) {
        rateLimitedUntilRef.current = Date.now() + RATE_LIMIT_COOLDOWN_MS;
        const msg = await response.text().catch(() => '');
        throw new Error('RATE_LIMITED:' + (msg || 'Please wait before scanning again.'));
      }

      let payload: ApiResponse<ScanResult> | null = null;
      try { payload = (await response.json()) as ApiResponse<ScanResult>; }
      catch { payload = null; }

      if (!response.ok || !payload?.success || !payload?.data) {
        throw new Error(payload?.message || 'Scan failed. Please try again.');
      }
      return payload.data;
    };

    const runLocalVisionScan = async (): Promise<ScanResult | null> => {
      for (let attempt = 1; attempt <= MAX_SCAN_RETRIES; attempt += 1) {
        try {
          // Check if we're in rate-limit cooldown
          if (Date.now() < rateLimitedUntilRef.current) {
            const waitSec = Math.ceil((rateLimitedUntilRef.current - Date.now()) / 1000);
            setError(`Vision server rate limited — waiting ${waitSec}s`);
            return null;
          }
          const result = await attemptScan();
          return result;
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Scan failed. Please try again.';
          // Don't retry on rate limit — just stop
          if (message.startsWith('RATE_LIMITED:')) {
            setError('Local vision rate limited — waiting before next scan.');
            speakAlert('Please wait a moment. Scanning too fast.', 'caution');
            return null;
          }
          if (attempt < MAX_SCAN_RETRIES) {
            setRetrying(true);
            setError(`Scan failed — retrying (${attempt}/${MAX_SCAN_RETRIES})`);
            await new Promise((resolve) => setTimeout(resolve, 3000));
            continue;
          }
          setError(message);
        }
      }
      return null;
    };

    const runLocalScan = async (): Promise<ScanResult | null> => {
      if (!modelLoaded) return null;
      try {
        return await performLocalScan();
      } catch (err) {
        console.error('Local detection failed:', err);
        return null;
      }
    };

    try {
      const useLocalVision = localVisionEnabledRef.current;
      if (useLocalVision) {
        const visionResult = await runLocalVisionScan();
        if (visionResult) {
          applyScanResult(visionResult, 'api');
          return true;
        }
      }

      const localResult = await runLocalScan();
      if (localResult) {
        applyScanResult(localResult, 'local');
        return true;
      }

      return false;
    } finally { finalizeScan(); }
  }, [applyScanResult, captureFrame, scanFrequency, speakAlert, startCamera, warmupCamera, modelLoaded, performLocalScan]);

  // ─── Auto scan ──────────────────────────────────────────────────────────
  const startAutoScan = useCallback(() => {
    stopAutoScan();
    void performScan();
    scanTimerRef.current = setInterval(() => {
      if (!isScanningRef.current) {
        void performScan();
      }
    }, scanFrequency);
  }, [performScan, scanFrequency, stopAutoScan]);

  useEffect(() => {
    if (cameraActiveRef.current && cameraReadyRef.current && !isPausedRef.current && !isManualScanRef.current) {
      startAutoScan();
    }
    return () => stopAutoScan();
  }, [scanFrequency, startAutoScan, stopAutoScan]);

  // ─── Manual scan trigger ────────────────────────────────────────────────
  const handleManualScan = useCallback(async () => {
    if (isScanningRef.current) return;
    if (!cameraReadyRef.current) {
      const warmed = await warmupCamera();
      if (!warmed) return;
    }
    isManualScanRef.current = true;
    isPausedRef.current = false;
    setIsManualScan(true);
    setIsPaused(false);
    stopAutoScan();
    await performScan();
    setIsManualScan(false);
    isManualScanRef.current = false;
    if (cameraActiveRef.current && !isPausedRef.current) { startAutoScan(); }
  }, [performScan, startAutoScan, stopAutoScan, warmupCamera]);

  // ─── Pause / Resume ─────────────────────────────────────────────────────
  const handlePauseResume = useCallback(() => {
    if (isPausedRef.current) {
      isPausedRef.current = false;
      setIsPaused(false);
      startAutoScan();
    } else {
      isPausedRef.current = true;
      setIsPaused(true);
      stopAutoScan();
    }
  }, [startAutoScan, stopAutoScan]);

  // ─── Camera start with auto-scan ────────────────────────────────────────
  const handleStart = useCallback(async () => {
    announceStatus('Starting camera and loading detection model.');
    // Start loading the model in parallel with camera
    loadModel();
    const started = await startCamera();
    if (started && cameraActiveRef.current) {
      const warmed = await warmupCamera();
      if (!warmed) return;
      // Give the camera a moment to stabilize before first scan
      announceStatus('Camera ready. Starting detection.');
      await new Promise((resolve) => setTimeout(resolve, 1500));
      if (cameraActiveRef.current) {
        startAutoScan();
      }
    }
  }, [announceStatus, loadModel, startAutoScan, startCamera, warmupCamera]);

  const handleStop = useCallback(() => {
    stopCamera();
    setCurrentResult(null);
    setScanCount(0);
    setScanHistory([]);
    setError(null);
  }, [stopCamera]);

  const handleVoiceCommand = useCallback((rawText: string) => {
    const normalized = rawText
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (!normalized) return;
    if (isSpeakingRef.current) return;

    const now = Date.now();
    if (
      normalized === lastVoiceCommandRef.current.text &&
      now - lastVoiceCommandRef.current.time < VOICE_COMMAND_DEBOUNCE_MS
    ) {
      return;
    }
    lastVoiceCommandRef.current = { text: normalized, time: now };

    const includesAny = (phrases: string[]) => phrases.some((phrase) => normalized.includes(phrase));

    const wantsStop = includesAny(['stop camera', 'stop scanning', 'stop scan', 'close camera', 'stop']);
    const wantsPause = includesAny(['pause scanning', 'pause scan', 'pause']);
    const wantsResume = includesAny(['resume scanning', 'resume scan', 'resume', 'continue']);
    const wantsStart = includesAny([
      'start camera',
      'open camera',
      'open app',
      'start app',
      'start navigation',
      'open nav assist',
      'start nav assist',
      'start assistant',
      'open assistant',
    ]);
    const wantsScan = includesAny(['scan now', 'scan', 'analyze', 'detect']);
    const wantsHealth = includesAny(['health check', 'check status', 'model status']);
    const wantsTestVoice = includesAny(['test voice', 'voice test']);

    if (wantsStop) {
      announceStatus('Voice command: stop camera.');
      handleStop();
      return;
    }
    if (wantsPause && !isPausedRef.current) {
      announceStatus('Voice command: pause scanning.');
      handlePauseResume();
      return;
    }
    if (wantsResume && isPausedRef.current) {
      announceStatus('Voice command: resume scanning.');
      handlePauseResume();
      return;
    }
    if (wantsStart) {
      announceStatus('Voice command: start camera.');
      if (!cameraActiveRef.current) {
        void handleStart();
      } else {
        void handleManualScan();
      }
      return;
    }
    if (wantsScan) {
      announceStatus('Voice command: scan now.');
      void handleManualScan();
      return;
    }
    if (wantsHealth) {
      announceStatus('Voice command: health check.');
      void handleHealthCheck();
      return;
    }
    if (wantsTestVoice) {
      announceStatus('Voice command: test voice.');
      testVoice();
    }
  }, [announceStatus, handleHealthCheck, handleManualScan, handlePauseResume, handleStart, handleStop, testVoice]);

  const stopVoiceRecognition = useCallback(() => {
    if (voiceRestartTimerRef.current) {
      clearTimeout(voiceRestartTimerRef.current);
      voiceRestartTimerRef.current = null;
    }
    const recognition = recognitionRef.current;
    if (!recognition) return;
    try { recognition.onresult = null; recognition.onstart = null; recognition.onend = null; recognition.onerror = null; } catch { return; }
    try { recognition.stop(); } catch { return; }
    try { recognition.abort(); } catch { return; }
    setVoiceListening(false);
  }, []);

  const startVoiceRecognition = useCallback(() => {
    const SpeechRecognitionCtor = getSpeechRecognitionCtor();
    if (!SpeechRecognitionCtor) {
      setVoiceError('Voice commands are not supported in this browser.');
      return;
    }

    let recognition = recognitionRef.current;
    if (!recognition) {
      recognition = new SpeechRecognitionCtor();
      recognition.lang = 'en-US';
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onresult = (event: any) => {
        const results = event?.results;
        if (!results || typeof results.length !== 'number') return;
        const last = results[results.length - 1];
        const transcript = last?.[0]?.transcript;
        if (!transcript) return;
        handleVoiceCommand(String(transcript));
      };

      recognition.onstart = () => {
        setVoiceListening(true);
        setVoiceError(null);
        setVoiceNeedsGesture(false);
      };

      recognition.onend = () => {
        setVoiceListening(false);
        if (!voiceCommandsEnabledRef.current) return;
        if (voiceRestartTimerRef.current) {
          clearTimeout(voiceRestartTimerRef.current);
        }
        voiceRestartTimerRef.current = window.setTimeout(() => {
          if (!voiceCommandsEnabledRef.current) return;
          try { recognition?.start(); } catch { return; }
        }, 700);
      };

      recognition.onerror = (event: any) => {
        const error = typeof event?.error === 'string' ? event.error : 'unknown';
        let message = 'Voice recognition error. Please try again.';
        if (error === 'not-allowed' || error === 'service-not-allowed') {
          message = 'Tap anywhere to enable voice commands.';
        } else if (error === 'audio-capture') {
          message = 'No microphone available for voice commands.';
        } else if (error === 'no-speech') {
          message = 'No speech detected.';
        }
        setVoiceError(message);
        announceStatus(message);
        setVoiceListening(false);
        if (error === 'not-allowed' || error === 'service-not-allowed') {
          setVoiceNeedsGesture(true);
        }
      };

      recognitionRef.current = recognition;
    }

    try {
      recognition.start();
      setVoiceError(null);
    } catch { return; }
  }, [announceStatus, getSpeechRecognitionCtor, handleVoiceCommand]);

  useEffect(() => {
    if (!mounted) return;
    if (!voiceCommandsEnabled) {
      stopVoiceRecognition();
      setVoiceListening(false);
      setVoiceNeedsGesture(false);
      return;
    }
    if (!voiceSupported) {
      setVoiceError('Voice commands are not supported in this browser.');
      setVoiceCommandsEnabled(false);
      return;
    }
    setVoiceError(null);
    startVoiceRecognition();
    return () => stopVoiceRecognition();
  }, [mounted, startVoiceRecognition, stopVoiceRecognition, voiceCommandsEnabled, voiceSupported]);

  useEffect(() => {
    if (!mounted || !voiceNeedsGesture || !voiceCommandsEnabled) return;
    const handleGesture = () => {
      if (!voiceCommandsEnabledRef.current) return;
      setVoiceNeedsGesture(false);
      startVoiceRecognition();
    };
    window.addEventListener('pointerdown', handleGesture, { once: true });
    window.addEventListener('keydown', handleGesture, { once: true });
    return () => {
      window.removeEventListener('pointerdown', handleGesture);
      window.removeEventListener('keydown', handleGesture);
    };
  }, [mounted, startVoiceRecognition, voiceCommandsEnabled, voiceNeedsGesture]);

  // ─── Helpers ────────────────────────────────────────────────────────────
  const positionLabel = (pos: string) => {
    switch (pos) {
      case 'ahead': return '⬆ Ahead';
      case 'center': return '⬆ Center';
      case 'left': return '⬅ Left';
      case 'right': return '➡ Right';
      default: return pos;
    }
  };

  const frequencyLabel = (ms: number) => {
    switch (ms) {
      case 4000: return '4s';
      case 6000: return '6s';
      case 10000: return '10s';
      default: return `${ms / 1000}s`;
    }
  };

  const threatLevelColor = (level: AlertLevel) => {
    switch (level) {
      case 'danger': return 'nav-threat-high';
      case 'caution': return 'nav-threat-medium';
      default: return 'nav-threat-low';
    }
  };

  // ─── Not mounted guard ──────────────────────────────────────────────────
  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <Eye className="h-10 w-10 text-primary nav-float" />
            <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-primary animate-ping" />
          </div>
          <p className="text-sm text-muted-foreground animate-pulse">Initializing NavAssist...</p>
        </div>
      </div>
    );
  }

  // ─── Render ─────────────────────────────────────────────────────────────
  const alertLevel = getAlertLevel(currentResult);
  const threatColor = alertLevel === 'danger'
    ? 'from-red-500/20 to-orange-500/10'
    : alertLevel === 'caution'
      ? 'from-yellow-500/15 to-amber-500/5'
      : 'from-emerald-500/10 to-cyan-500/5';

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <ARIAStatus message={ariaStatus} />

      {/* ─── Header ─────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 nav-glass-strong border-b border-border/50">
        <div className="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="p-2 rounded-xl bg-primary/10 border border-primary/20">
                <Eye className="h-6 w-6 text-primary" />
              </div>
              {cameraActive && (
                <span className="absolute -top-0.5 -right-0.5 h-3 w-3 rounded-full bg-emerald-500">
                  <span className="absolute inset-0 rounded-full bg-emerald-500 animate-ping opacity-75" />
                </span>
              )}
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight leading-none nav-text-gradient">
                NavAssist
              </h1>
              <p className="text-[11px] text-muted-foreground mt-0.5">
                AI Navigation Assistant
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {cameraActive && (
              <Badge
                variant="outline"
                className="text-xs font-mono gap-1.5 border-primary/30 bg-primary/5"
              >
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-500 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                {scanCount} scans
              </Badge>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label="Toggle theme"
              className="rounded-xl hover:bg-primary/10"
            >
              {theme === 'dark' ? (
                <Sun className="h-5 w-5" />
              ) : (
                <Moon className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>
      </header>

      {/* ─── Main ───────────────────────────────────────────────────── */}
      <main
        className="flex-1 max-w-2xl w-full mx-auto px-4 py-7 space-y-6"
        aria-busy={isScanning}
      >
        {/* ─── Hero ─────────────────────────────────────────────── */}
        <section className="nav-hero px-5 py-6 sm:px-7">
          <div className="nav-hero-content space-y-4">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-2">
                <p className="text-[11px] uppercase tracking-[0.4em] text-primary/80">
                  Mobility Briefing
                </p>
                <h2 className="text-3xl sm:text-4xl font-semibold leading-tight">
                  Clear guidance, calm delivery, safer movement.
                </h2>
                <p className="text-sm text-muted-foreground max-w-md">
                  Continuous scene understanding with concise voice prompts and haptic cues.
                </p>
              </div>
              <div className="flex items-center gap-2 rounded-full nav-hero-badge px-3 py-2">
                <span className={`nav-hero-glow ${cameraActive ? '' : 'opacity-40'}`} />
                <span className="text-[11px] font-mono uppercase tracking-[0.25em]">
                  {cameraActive ? 'Live' : 'Standby'}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              <Badge className="nav-hero-badge text-[10px] font-mono uppercase tracking-widest">
                {cameraActive ? (cameraReady ? 'Camera Live' : 'Camera Warming') : 'Camera Offline'}
              </Badge>
              <Badge className="nav-hero-badge text-[10px] font-mono uppercase tracking-widest">
                Mode: {detectionMode === 'local' ? 'Local AI' : 'Local VLM'}
              </Badge>
              <Badge className="nav-hero-badge text-[10px] font-mono uppercase tracking-widest">
                Auto Scan: {frequencyLabel(scanFrequency)}
              </Badge>
              <Badge className="nav-hero-badge text-[10px] font-mono uppercase tracking-widest">
                Haptics: {hapticEnabled ? 'On' : 'Off'}
              </Badge>
              {modelLoading && (
                <Badge className="nav-hero-badge text-[10px] font-mono uppercase tracking-widest">
                  Local Model Loading
                </Badge>
              )}
              {modelError && (
                <Badge className="nav-hero-badge text-[10px] font-mono uppercase tracking-widest text-red-300">
                  Local Model Error
                </Badge>
              )}
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <div className="nav-feature-card px-4 py-3">
                <div className="relative flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.25em] text-primary/80">
                  <Mic className="h-3.5 w-3.5" /> Hands-free
                </div>
                <p className="relative mt-2 text-sm text-foreground/80">
                  Voice commands and calm alerts when your hands are busy.
                </p>
              </div>
              <div className="nav-feature-card px-4 py-3">
                <div className="relative flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.25em] text-primary/80">
                  <ShieldCheck className="h-3.5 w-3.5" /> Safety-first
                </div>
                <p className="relative mt-2 text-sm text-foreground/80">
                  Prioritized hazards and direction cues for quick decisions.
                </p>
              </div>
              <div className="nav-feature-card px-4 py-3">
                <div className="relative flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.25em] text-primary/80">
                  <Zap className="h-3.5 w-3.5" /> Adaptive AI
                </div>
                <p className="relative mt-2 text-sm text-foreground/80">
                  Local detection with optional SmolVLM2 server for higher accuracy.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ─── Camera View ─────────────────────────────────────────── */}
        <div className={`nav-camera-container ${isScanning ? 'nav-scan-active' : ''} ${
          alertLevel === 'danger' ? 'nav-danger-glow' :
          alertLevel === 'caution' ? 'nav-caution-glow' :
          cameraActive ? 'nav-glow' : ''
        }`}>
          <div className="relative bg-black aspect-video w-full overflow-hidden rounded-2xl">
            {/* Camera feed */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              aria-label="Live camera feed"
              aria-hidden={!cameraActive || Boolean(cameraError)}
              className="w-full h-full object-cover"
            />

            {/* Placeholder when camera is off */}
            {(cameraError || !cameraActive) && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-gradient-to-b from-black/80 to-black/95">
                <div className="p-5 rounded-full bg-white/5 border border-white/10">
                  <CameraOff className="h-10 w-10 text-white/40" />
                </div>
                <div className="text-center px-8">
                  <p className="text-sm text-white/50 font-medium">
                    {cameraError || 'Camera is not active'}
                  </p>
                  {!cameraError && (
                    <p className="text-xs text-white/30 mt-1">
                      Press &quot;Start Camera&quot; to begin scanning
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Scanning sweep line */}
            {cameraActive && isScanning && (
              <>
                <div className="absolute inset-0 pointer-events-none">
                  <div
                    className="nav-scan-line absolute left-0 right-0 h-0.5"
                    style={{
                      background: 'linear-gradient(90deg, transparent, oklch(0.62 0.17 175 / 80%), transparent)',
                      boxShadow: '0 0 20px oklch(0.62 0.17 175 / 40%)',
                    }}
                  />
                </div>
                <div className="absolute inset-0 border-2 border-primary/30 rounded-2xl pointer-events-none animate-pulse" />
              </>
            )}

            {/* Threat level banner */}
            {currentResult && alertLevel !== 'clear' && (
              <div
                className={`absolute top-0 left-0 right-0 px-4 py-2.5 text-center font-semibold text-sm flex items-center justify-center gap-2 backdrop-blur-md ${
                  alertLevel === 'danger'
                    ? 'bg-red-600/85 text-white'
                    : 'bg-amber-500/85 text-black'
                }`}
              >
                {alertLevel === 'danger' ? (
                  <AlertTriangle className="h-5 w-5 animate-pulse" />
                ) : (
                  <Shield className="h-5 w-5" />
                )}
                {alertLevel === 'danger'
                  ? '⚠ DANGER — Immediate hazard detected'
                  : '⚠ CAUTION — Potential hazard ahead'}
              </div>
            )}

            {/* Status badge */}
            {cameraActive && (
              <div className="absolute bottom-3 right-3">
                <Badge
                  variant="secondary"
                  className="text-xs font-mono backdrop-blur-xl bg-black/60 text-white border border-white/10 gap-1.5 px-3 py-1"
                >
                  <span
                    className={`h-2 w-2 rounded-full ${
                      isScanning
                        ? 'bg-primary animate-pulse'
                        : isPaused || !cameraReady
                          ? 'bg-amber-400'
                          : 'bg-emerald-500'
                    }`}
                  />
                  {isScanning
                    ? 'Analyzing…'
                    : isPaused
                      ? 'Paused'
                      : cameraReady
                        ? 'Live'
                        : 'Warming up'}
                </Badge>
              </div>
            )}

            {/* Error banner */}
            {error && (
              <div className="absolute bottom-3 left-3 right-16">
                <Badge
                  variant="destructive"
                  className="text-xs backdrop-blur-md gap-1.5 bg-red-600/80 border-0"
                >
                  <RefreshCw className={`h-3 w-3 ${retrying ? 'animate-spin' : ''}`} />
                  {error}
                </Badge>
              </div>
            )}
          </div>
        </div>

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* ─── Voice Alert ─────────────────────────────────────────── */}
        {currentResult?.action_advice && (
          <div className={`nav-glass nav-voice-panel rounded-2xl px-6 py-5 transition-all duration-500 bg-gradient-to-r ${threatColor}`}>
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-xl bg-primary/10 border border-primary/20 shrink-0 mt-0.5">
                {isSpeaking ? (
                  <div className="nav-voice-wave">
                    <span /><span /><span /><span /><span />
                  </div>
                ) : (
                  <Volume2 className="h-5 w-5 text-primary" />
                )}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-[10px] font-semibold mb-1.5 uppercase tracking-widest text-primary/80">
                  Action Advice
                </p>
                <p className="text-base font-semibold leading-snug text-foreground">
                  &ldquo;{currentResult.action_advice}&rdquo;
                </p>
                {currentResult.scene_summary && (
                  <div className="mt-2 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-xs font-medium text-primary">
                    <Navigation className="h-3 w-3" />
                    {currentResult.scene_summary}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ─── Threat List ─────────────────────────────────────────── */}
        {currentResult && (
          <div className="nav-glass rounded-2xl overflow-hidden nav-gradient-border">
            <div className="px-5 py-3 border-b border-border/30 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                {alertLevel === 'danger' ? (
                  <ShieldAlert className="h-5 w-5 text-red-400" />
                ) : alertLevel === 'caution' ? (
                  <Shield className="h-5 w-5 text-amber-400" />
                ) : (
                  <ShieldCheck className="h-5 w-5 text-emerald-400" />
                )}
                <span className="text-sm font-semibold">Detected Objects</span>
                <Badge variant="outline" className="text-[10px] font-mono">
                  {currentResult.objects.length}
                </Badge>
              </div>
            </div>
            <div className="p-4">
              {currentResult.objects.length === 0 ? (
                <div className="flex items-center gap-3 py-3 px-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                  <ShieldCheck className="h-5 w-5 text-emerald-400" />
                  <div>
                    <p className="text-sm font-medium text-emerald-400">Path is Clear</p>
                    <p className="text-xs text-muted-foreground mt-0.5">No obstacles detected — safe to proceed</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                  {currentResult.objects.map((obj, i) => {
                    const urgency = objectUrgency(obj.distance);
                    return (
                    <div
                      key={i}
                      className={`flex items-center justify-between rounded-xl px-4 py-3 transition-all duration-300 ${
                        urgency === 'high'
                          ? 'bg-red-500/8 border border-red-500/15 hover:bg-red-500/12'
                          : urgency === 'medium'
                            ? 'bg-amber-500/8 border border-amber-500/15 hover:bg-amber-500/12'
                            : 'bg-foreground/5 border border-foreground/10 hover:bg-foreground/8'
                      }`}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className={`p-1.5 rounded-lg ${
                          urgency === 'high' ? 'bg-red-500/15' :
                          urgency === 'medium' ? 'bg-amber-500/15' : 'bg-emerald-500/15'
                        }`}>
                          <Scan className={`h-4 w-4 ${
                            urgency === 'high'
                              ? 'text-red-400 animate-pulse'
                              : urgency === 'medium'
                                ? 'text-amber-400 nav-float'
                                : 'text-emerald-400'
                          }`} />
                        </div>
                        <span className="text-sm font-medium truncate">
                          {obj.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0 ml-3">
                        <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md ${threatLevelColor(urgency === 'high' ? 'danger' : urgency === 'medium' ? 'caution' : 'clear')}`}>
                          {obj.distance}
                        </span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-foreground/5 text-muted-foreground">
                          {positionLabel(obj.direction)}
                        </span>
                      </div>
                    </div>
                  );})}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ─── Scene Summary ───────────────────────────────────────── */}
        {currentResult?.scene_summary && (
          <div className="nav-glass rounded-2xl px-5 py-3.5 nav-shimmer">
            <p className="text-sm text-muted-foreground leading-relaxed">
              <span className="font-semibold text-foreground mr-1">📍 Scene:</span>
              {currentResult.scene_summary}
            </p>
          </div>
        )}

        {/* ─── Controls ────────────────────────────────────────────── */}
        <div className="nav-glass rounded-2xl p-5 nav-gradient-border">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {/* Start / Stop Camera */}
            {!cameraActive ? (
              <Button
                ref={startButtonRef}
                onClick={handleStart}
                className="col-span-2 sm:col-span-1 gap-2 h-14 rounded-xl nav-btn-glow bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-base"
                size="lg"
                aria-label="Start camera"
              >
                <Camera className="h-5 w-5" />
                Start Camera
              </Button>
            ) : (
              <Button
                onClick={handleStop}
                variant="destructive"
                className="col-span-2 sm:col-span-1 gap-2 h-14 rounded-xl nav-btn-glow font-semibold text-base"
                size="lg"
                aria-label="Stop camera"
              >
                <CameraOff className="h-5 w-5" />
                Stop
              </Button>
            )}

            {/* Manual Scan */}
            <Button
              ref={scanButtonRef}
              onClick={handleManualScan}
              disabled={!cameraActive || !cameraReady || isScanning}
              variant="outline"
              className="col-span-1 gap-2 h-14 rounded-xl nav-btn-glow border-primary/30 hover:bg-primary/10 font-semibold text-base"
              size="lg"
              aria-label="Scan now"
            >
              <Eye className={`h-5 w-5 ${isScanning ? 'animate-pulse' : ''}`} />
              Scan Now
            </Button>

            {/* Pause / Resume */}
            <Button
              onClick={handlePauseResume}
              disabled={!cameraActive}
              variant={isPaused ? 'default' : 'outline'}
              className={`col-span-1 gap-2 h-14 rounded-xl nav-btn-glow font-semibold text-base ${
                isPaused ? '' : 'border-primary/30 hover:bg-primary/10'
              }`}
              size="lg"
              aria-pressed={isPaused}
              aria-label={isPaused ? 'Resume auto scanning' : 'Pause auto scanning'}
            >
              {isPaused ? (
                <><Play className="h-5 w-5" /> Resume</>
              ) : (
                <><Pause className="h-5 w-5" /> Pause</>
              )}
            </Button>

            {/* Frequency Selector */}
            <div
              className="col-span-2 sm:col-span-1 flex items-center gap-2"
              role="group"
              aria-label="Scan frequency"
            >
              <div className="flex rounded-xl border border-primary/20 overflow-hidden w-full bg-foreground/5">
                {SCAN_FREQUENCIES.map((freq) => (
                  <button
                    key={freq}
                    type="button"
                    onClick={() => setScanFrequency(freq)}
                    aria-label={`Set scan frequency to ${frequencyLabel(freq)}`}
                    aria-pressed={scanFrequency === freq}
                    className={`flex-1 px-3 py-4 text-sm font-mono font-semibold transition-all duration-300 ${
                      scanFrequency === freq
                        ? 'bg-primary text-primary-foreground shadow-lg'
                        : 'text-muted-foreground hover:bg-primary/10 hover:text-foreground'
                    }`}
                  >
                    {frequencyLabel(freq)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Secondary controls */}
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Button
              onClick={testVoice}
              variant="outline"
              className="gap-2 h-12 rounded-xl nav-btn-glow border-primary/20 hover:bg-primary/10"
              size="lg"
              aria-label="Test voice output"
            >
              <Volume2 className="h-4 w-4" />
              Test Voice
            </Button>

            <Button
              onClick={handleHealthCheck}
              variant="outline"
              className="gap-2 h-12 rounded-xl nav-btn-glow border-primary/20 hover:bg-primary/10"
              size="lg"
              aria-label="Check vision model status"
              disabled={healthChecking}
            >
              <Activity className={healthChecking ? 'h-4 w-4 animate-spin' : 'h-4 w-4'} />
              {healthChecking ? 'Checking…' : 'Health Check'}
            </Button>
          </div>

          {/* Settings */}
          <div className="mt-4 space-y-3">
            <div className="flex items-center justify-between rounded-xl border border-primary/15 bg-primary/3 px-4 py-3">
              <div className="space-y-0.5">
                <Label htmlFor="local-vision-toggle" className="text-xs font-semibold flex items-center gap-2">
                  <Zap className="h-3.5 w-3.5 text-primary" />
                  Local SmolVLM2 (Offline)
                </Label>
                <p className="text-[10px] text-muted-foreground">
                  Higher accuracy when the local SmolVLM2 server is running.
                </p>
              </div>
              <Switch
                id="local-vision-toggle"
                checked={localVisionEnabled}
                onCheckedChange={setLocalVisionEnabled}
                aria-label="Toggle local SmolVLM2 vision"
              />
            </div>

            <div className="flex items-center justify-between rounded-xl border border-primary/15 bg-primary/3 px-4 py-3">
              <div className="space-y-0.5">
                <Label htmlFor="voice-toggle" className="text-xs font-semibold flex items-center gap-2">
                  <Mic className="h-3.5 w-3.5 text-primary" />
                  Voice Commands
                </Label>
                <p className="text-[10px] text-muted-foreground">
                  {!voiceSupported
                    ? 'Not supported in this browser.'
                    : voiceNeedsGesture
                      ? 'Tap anywhere to enable voice commands.'
                      : voiceListening
                        ? 'Listening: say "start camera", "scan now", "stop camera".'
                        : 'Say: "start camera", "scan now", "stop camera".'}
                </p>
              </div>
              <Switch
                id="voice-toggle"
                checked={voiceCommandsEnabled}
                onCheckedChange={setVoiceCommandsEnabled}
                aria-label="Toggle voice commands"
                disabled={!voiceSupported}
              />
            </div>

            <div className="flex items-center justify-between rounded-xl border border-primary/15 bg-primary/3 px-4 py-3">
              <div className="space-y-0.5">
                <Label htmlFor="haptic-toggle" className="text-xs font-semibold flex items-center gap-2">
                  <Radio className="h-3.5 w-3.5 text-primary" />
                  Haptic Alerts
                </Label>
                <p className="text-[10px] text-muted-foreground">
                  Vibrations for scan results
                </p>
              </div>
              <Switch
                id="haptic-toggle"
                checked={hapticEnabled}
                onCheckedChange={setHapticEnabled}
                aria-label="Toggle haptic feedback"
              />
            </div>

            {cameraDevices.length > 1 && (
              <div className="flex items-center justify-between rounded-xl border border-primary/15 bg-primary/3 px-4 py-3">
                <div className="space-y-0.5">
                  <Label htmlFor="camera-picker" className="text-xs font-semibold flex items-center gap-2">
                    <Camera className="h-3.5 w-3.5 text-primary" />
                    Camera Source
                  </Label>
                  <p className="text-[10px] text-muted-foreground">
                    Choose the active camera
                  </p>
                </div>
                <select
                  id="camera-picker"
                  value={selectedDeviceId}
                  onChange={(event) => {
                    const nextId = event.target.value;
                    setSelectedDeviceId(nextId);
                    window.localStorage.setItem(CAMERA_STORAGE_KEY, nextId);
                  }}
                  className="rounded-lg border border-primary/20 bg-background/50 px-3 py-2 text-sm text-foreground backdrop-blur"
                  aria-label="Select camera device"
                >
                  {cameraDevices.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${index + 1}`}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {healthStatus && (
              <div className={`rounded-xl border px-4 py-2.5 text-xs font-medium ${
                healthStatus.includes('✓')
                  ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-400'
                  : healthStatus.includes('✗')
                    ? 'border-red-500/20 bg-red-500/5 text-red-400'
                    : 'border-primary/15 bg-primary/5 text-muted-foreground'
              }`}>
                {healthStatus}
              </div>
            )}

            {voiceError && (
              <div className="rounded-xl border border-red-500/20 bg-red-500/5 px-4 py-2.5 text-xs font-medium text-red-400">
                {voiceError}
              </div>
            )}
          </div>
        </div>

        {/* ─── Scan History ────────────────────────────────────────── */}
        {scanHistory.length > 1 && (
          <div className="nav-glass rounded-2xl overflow-hidden">
            <div className="px-5 py-3 border-b border-border/30 flex items-center gap-2">
              <RotateCcw className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">Recent Scans</span>
              <Badge variant="outline" className="text-[10px] font-mono border-primary/30">
                {scanHistory.length}
              </Badge>
            </div>
            <div className="p-4 space-y-2 max-h-48 overflow-y-auto nav-timeline">
              {scanHistory.slice(1).map((scan, i) => (
                <div
                  key={i}
                  className="nav-timeline-item flex items-center gap-3 text-xs py-2 px-3 rounded-xl bg-foreground/5 border border-foreground/10 hover:bg-foreground/8 transition-colors"
                >
                  <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${threatLevelColor(getAlertLevel(scan))}`}>
                    {getAlertLevel(scan)}
                  </span>
                  <span className="truncate flex-1 text-muted-foreground">{scan.action_advice || scan.scene_summary}</span>
                  {scan.objects.length > 0 && (
                    <Badge variant="secondary" className="text-[10px] shrink-0 bg-primary/10 text-primary border-0">
                      {scan.objects.length} item{scan.objects.length > 1 ? 's' : ''}
                    </Badge>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── Camera Error (full page) ───────────────────────────── */}
        {cameraError && !cameraActive && (
          <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-5">
            <div className="flex items-start gap-3">
              <div className="p-2 rounded-xl bg-red-500/10 border border-red-500/20 shrink-0">
                <AlertTriangle className="h-5 w-5 text-red-400" />
              </div>
              <div>
                <h3 className="font-semibold text-sm text-red-400 mb-1">
                  Camera Access Required
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {cameraError}
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ─── Footer ────────────────────────────────────────────────── */}
      <footer className="border-t border-border/30 nav-glass-strong px-4 py-4 text-center text-xs text-muted-foreground">
        <p className="flex items-center justify-center gap-2">
          <Eye className="h-3.5 w-3.5 text-primary" />
          <span className="nav-text-gradient font-semibold">NavAssist</span>
          <span className="opacity-50">—</span>
          AI-Powered Blind Navigation Assistant
        </p>
        <p className="mt-1 opacity-50">
          For accessibility assistance only. Not a replacement for mobility aids.
        </p>
      </footer>
    </div>
  );
}
