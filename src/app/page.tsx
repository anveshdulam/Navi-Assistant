'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useTheme } from 'next-themes';
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
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Threat {
  label: string;
  urgency: 'high' | 'medium' | 'low';
  position: 'ahead' | 'left' | 'right';
}

interface ScanResult {
  threats: Threat[];
  voiceAlert: string;
  safeAction: string;
  summary: string;
  threatLevel: 'clear' | 'caution' | 'danger';
}

type ScanFrequency = 4000 | 6000 | 10000;

// ─── Theme variables ────────────────────────────────────────────────────────

const themeVars = `
  :root {
    --nav-camera-border: oklch(0.145 0 0 / 8%);
    --nav-scan-pulse: #22c55e;
    --nav-danger-bg: oklch(0.577 0.245 27.325 / 90%);
    --nav-danger-text: white;
    --nav-caution-bg: oklch(0.795 0.184 86.047 / 90%);
    --nav-caution-text: oklch(0.205 0 0);
    --nav-voice-bg: oklch(0.541 0.281 293.009 / 12%);
    --nav-voice-text: oklch(0.541 0.281 293.009);
    --nav-voice-border: oklch(0.541 0.281 293.009 / 25%);
    --nav-high-badge-bg: oklch(0.577 0.245 27.325 / 15%);
    --nav-high-badge-text: oklch(0.577 0.245 27.325);
    --nav-medium-badge-bg: oklch(0.795 0.184 86.047 / 15%);
    --nav-medium-badge-text: oklch(0.795 0.184 86.047);
    --nav-low-badge-bg: oklch(0.648 0.2 145.947 / 15%);
    --nav-low-badge-text: oklch(0.648 0.2 145.947);
    --nav-footer-bg: oklch(0.97 0 0);
    --nav-footer-text: oklch(0.556 0 0);
    --nav-footer-border: oklch(0.922 0 0);
  }

  .dark {
    --nav-camera-border: oklch(1 0 0 / 10%);
    --nav-scan-pulse: #22c55e;
    --nav-danger-bg: oklch(0.704 0.191 22.216 / 90%);
    --nav-danger-text: white;
    --nav-caution-bg: oklch(0.795 0.184 86.047 / 90%);
    --nav-caution-text: oklch(0.145 0 0);
    --nav-voice-bg: oklch(0.541 0.281 293.009 / 15%);
    --nav-voice-text: oklch(0.723 0.219 292.581);
    --nav-voice-border: oklch(0.541 0.281 293.009 / 30%);
    --nav-high-badge-bg: oklch(0.704 0.191 22.216 / 15%);
    --nav-high-badge-text: oklch(0.704 0.191 22.216);
    --nav-medium-badge-bg: oklch(0.795 0.184 86.047 / 15%);
    --nav-medium-badge-text: oklch(0.795 0.184 86.047);
    --nav-low-badge-bg: oklch(0.648 0.2 145.947 / 15%);
    --nav-low-badge-text: oklch(0.648 0.2 145.947);
    --nav-footer-bg: oklch(0.205 0 0);
    --nav-footer-text: oklch(0.708 0 0);
    --nav-footer-border: oklch(1 0 0 / 8%);
  }
`;

// ─── Component ───────────────────────────────────────────────────────────────

export default function BlindNavigationAssistant() {
  // Theme
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // Camera
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Scanning
  const [isScanning, setIsScanning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [scanFrequency, setScanFrequency] = useState<ScanFrequency>(6000);
  const [scanCount, setScanCount] = useState(0);
  const [isManualScan, setIsManualScan] = useState(false);
  const scanTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isScanningRef = useRef(false);

  // Results
  const [currentResult, setCurrentResult] = useState<ScanResult | null>(null);
  const [scanHistory, setScanHistory] = useState<ScanResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState(false);
  const retryCountRef = useRef(0);

  // Animation
  const [isSpeaking, setIsSpeaking] = useState(false);

  // ─── Initialize ─────────────────────────────────────────────────────────

  useEffect(() => {
    setMounted(true);
    return () => {
      stopCamera();
      stopAutoScan();
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  // ─── Theme toggle ───────────────────────────────────────────────────────

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  }, [theme, setTheme]);

  // ─── Camera ─────────────────────────────────────────────────────────────

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      const message =
        err instanceof DOMException
          ? err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError'
            ? 'Camera access denied. Please grant camera permission in your browser settings and reload the page.'
            : `Camera error: ${err.message}`
          : 'Failed to access camera. Please ensure your device has a camera and try again.';
      setCameraError(message);
      setCameraActive(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsScanning(false);
    setIsPaused(false);
    stopAutoScan();
  }, []);

  // ─── Capture frame ──────────────────────────────────────────────────────

  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Use a smaller resolution for faster upload
    const maxWidth = 1024;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
  }, []);

  // ─── Vision scan ────────────────────────────────────────────────────────

  const performScan = useCallback(async () => {
    if (isScanningRef.current) return;
    isScanningRef.current = true;
    setIsScanning(true);
    setError(null);

    try {
      const frame = captureFrame();
      if (!frame) {
        throw new Error('Could not capture frame from camera');
      }

      const response = await fetch('/api/vision-scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: frame }),
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();
      if (!data.success || !data.data) {
        throw new Error(data.error || 'Invalid response');
      }

      const result: ScanResult = data.data;
      setCurrentResult(result);
      setScanHistory((prev) => [result, ...prev].slice(0, 10));
      setScanCount((c) => c + 1);
      retryCountRef.current = 0;

      // Speak the voice alert
      speakAlert(result.voiceAlert, result.threatLevel);
    } catch (err) {
      retryCountRef.current += 1;
      if (retryCountRef.current <= 3) {
        setRetrying(true);
        setError(`Scan failed — retrying (${retryCountRef.current}/3)`);
        setTimeout(() => {
          setRetrying(false);
          isScanningRef.current = false;
        }, 1500);
      } else {
        setError('Scan failed — retrying');
        retryCountRef.current = 0;
      }
    } finally {
      if (retryCountRef.current === 0) {
        isScanningRef.current = false;
        setIsScanning(false);
      }
    }
  }, [captureFrame]);

  // ─── TTS ────────────────────────────────────────────────────────────────

  const speakAlert = useCallback(
    (text: string, threatLevel: string) => {
      if (!('speechSynthesis' in window)) return;

      // Cancel any ongoing speech
      window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);

      // Adjust speed and pitch based on threat level
      switch (threatLevel) {
        case 'danger':
          utterance.rate = 1.3;
          utterance.pitch = 1.4;
          utterance.volume = 1.0;
          break;
        case 'caution':
          utterance.rate = 1.1;
          utterance.pitch = 1.1;
          utterance.volume = 0.9;
          break;
        default:
          utterance.rate = 0.95;
          utterance.pitch = 1.0;
          utterance.volume = 0.8;
          break;
      }

      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    },
    []
  );

  const testVoice = useCallback(() => {
    speakAlert('Voice test successful. Navigation assistant is ready.', 'clear');
  }, [speakAlert]);

  // ─── Auto scan ──────────────────────────────────────────────────────────

  const startAutoScan = useCallback(() => {
    stopAutoScan();
    performScan();
    scanTimerRef.current = setInterval(() => {
      if (!isScanningRef.current) {
        performScan();
      }
    }, scanFrequency);
  }, [performScan, scanFrequency]);

  const stopAutoScan = useCallback(() => {
    if (scanTimerRef.current) {
      clearInterval(scanTimerRef.current);
      scanTimerRef.current = null;
    }
  }, []);

  // Restart timer when frequency changes
  useEffect(() => {
    if (cameraActive && !isPaused && !isManualScan) {
      startAutoScan();
    }
    return () => stopAutoScan();
  }, [scanFrequency]);

  // ─── Manual scan trigger ────────────────────────────────────────────────

  const handleManualScan = useCallback(() => {
    setIsManualScan(true);
    setIsPaused(false);
    stopAutoScan();
    performScan();
    // Resume auto-scan after manual scan completes
    setTimeout(() => {
      setIsManualScan(false);
      if (cameraActive && !isPaused) {
        startAutoScan();
      }
    }, 2000);
  }, [performScan, cameraActive, isPaused, startAutoScan]);

  // ─── Pause / Resume ─────────────────────────────────────────────────────

  const handlePauseResume = useCallback(() => {
    if (isPaused) {
      setIsPaused(false);
      startAutoScan();
    } else {
      setIsPaused(true);
      stopAutoScan();
    }
  }, [isPaused, startAutoScan, stopAutoScan]);

  // ─── Camera start with auto-scan ────────────────────────────────────────

  const handleStart = useCallback(async () => {
    await startCamera();
    // Small delay for camera to initialize
    setTimeout(() => {
      startAutoScan();
    }, 1000);
  }, [startCamera, startAutoScan]);

  const handleStop = useCallback(() => {
    stopCamera();
    setCurrentResult(null);
    setScanCount(0);
    setScanHistory([]);
    setError(null);
  }, [stopCamera]);

  // ─── Helpers ────────────────────────────────────────────────────────────

  const urgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'high':
        return 'destructive';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'secondary';
    }
  };

  const positionLabel = (pos: string) => {
    switch (pos) {
      case 'ahead':
        return '⬆ Ahead';
      case 'left':
        return '⬅ Left';
      case 'right':
        return '➡ Right';
      default:
        return pos;
    }
  };

  const frequencyLabel = (ms: number) => {
    switch (ms) {
      case 4000:
        return '4s';
      case 6000:
        return '6s';
      case 10000:
        return '10s';
      default:
        return `${ms / 1000}s`;
    }
  };

  const threatIcon = (level: string) => {
    switch (level) {
      case 'danger':
        return <ShieldAlert className="h-5 w-5 text-[var(--nav-high-badge-text)]" />;
      case 'caution':
        return <Shield className="h-5 w-5 text-[var(--nav-medium-badge-text)]" />;
      default:
        return <ShieldCheck className="h-5 w-5 text-[var(--nav-low-badge-text)]" />;
    }
  };

  // ─── Not mounted guard ──────────────────────────────────────────────────

  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Activity className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  // ─── Render ─────────────────────────────────────────────────────────────

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: themeVars }} />

      <div className="min-h-screen flex flex-col bg-background text-foreground">
        {/* ─── Header ─────────────────────────────────────────────────── */}
        <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
          <div className="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Eye className="h-7 w-7 text-primary" />
                <span className="absolute -top-0.5 -right-0.5 h-3 w-3 rounded-full bg-[var(--nav-scan-pulse)] animate-pulse" />
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight leading-none">
                  NavAssist
                </h1>
                <p className="text-xs text-muted-foreground">
                  Blind Navigation Assistant
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {cameraActive && (
                <Badge
                  variant="outline"
                  className="text-xs font-mono gap-1.5"
                >
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--nav-scan-pulse)] opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--nav-scan-pulse)]" />
                  </span>
                  Scans: {scanCount}
                </Badge>
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                aria-label="Toggle theme"
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
        <main className="flex-1 max-w-2xl w-full mx-auto px-4 py-6 space-y-5">
          {/* ─── Camera View ─────────────────────────────────────────── */}
          <Card className="overflow-hidden border-border">
            <div className="relative bg-black aspect-video w-full">
              {/* Camera feed */}
              {cameraActive && !cameraError ? (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-white/60">
                  <CameraOff className="h-12 w-12" />
                  <p className="text-sm text-center px-4">
                    {cameraError || 'Camera is not active'}
                  </p>
                </div>
              )}

              {/* Scanning indicator overlay */}
              {cameraActive && isScanning && (
                <div className="absolute inset-0 border-4 border-[var(--nav-scan-pulse)]/30 rounded-none animate-pulse pointer-events-none" />
              )}

              {/* Threat level banner */}
              {currentResult && currentResult.threatLevel !== 'clear' && (
                <div
                  className="absolute top-0 left-0 right-0 px-4 py-2.5 text-center font-semibold text-sm flex items-center justify-center gap-2"
                  style={{
                    backgroundColor:
                      currentResult.threatLevel === 'danger'
                        ? 'var(--nav-danger-bg)'
                        : 'var(--nav-caution-bg)',
                    color:
                      currentResult.threatLevel === 'danger'
                        ? 'var(--nav-danger-text)'
                        : 'var(--nav-caution-text)',
                  }}
                >
                  {currentResult.threatLevel === 'danger' ? (
                    <AlertTriangle className="h-5 w-5" />
                  ) : (
                    <Shield className="h-5 w-5" />
                  )}
                  {currentResult.threatLevel === 'danger'
                    ? '⚠ DANGER — Immediate hazard detected'
                    : '⚠ CAUTION — Potential hazard ahead'}
                </div>
              )}

              {/* Scanning status badge */}
              {cameraActive && (
                <div className="absolute bottom-3 right-3">
                  <Badge
                    variant="secondary"
                    className="text-xs font-mono backdrop-blur-sm bg-black/50 text-white border-0 gap-1.5"
                  >
                    <span
                      className={`h-2 w-2 rounded-full ${
                        isScanning
                          ? 'bg-[var(--nav-scan-pulse)] animate-pulse'
                          : isPaused
                            ? 'bg-yellow-400'
                            : 'bg-[var(--nav-scan-pulse)]'
                      }`}
                    />
                    {isScanning
                      ? 'Scanning…'
                      : isPaused
                        ? 'Paused'
                        : 'Active'}
                  </Badge>
                </div>
              )}

              {/* Error banner */}
              {error && (
                <div className="absolute bottom-3 left-3 right-16">
                  <Badge
                    variant="destructive"
                    className="text-xs backdrop-blur-sm gap-1"
                  >
                    <RefreshCw className={`h-3 w-3 ${retrying ? 'animate-spin' : ''}`} />
                    {error}
                  </Badge>
                </div>
              )}
            </div>

            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} className="hidden" />
          </Card>

          {/* ─── Voice Alert (Purple Highlighted) ────────────────────── */}
          {currentResult?.voiceAlert && (
            <div
              className="rounded-xl border px-5 py-4 transition-all duration-300"
              style={{
                backgroundColor: 'var(--nav-voice-bg)',
                borderColor: 'var(--nav-voice-border)',
              }}
            >
              <div className="flex items-start gap-3">
                <Volume2
                  className={`h-5 w-5 mt-0.5 shrink-0 ${
                    isSpeaking ? 'animate-pulse' : ''
                  }`}
                  style={{ color: 'var(--nav-voice-text)' }}
                />
                <div className="min-w-0">
                  <p className="text-xs font-medium mb-1 uppercase tracking-wider" style={{ color: 'var(--nav-voice-text)' }}>
                    Voice Alert
                  </p>
                  <p className="text-base font-semibold leading-snug" style={{ color: 'var(--nav-voice-text)' }}>
                    &ldquo;{currentResult.voiceAlert}&rdquo;
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* ─── Threat List ─────────────────────────────────────────── */}
          {currentResult && (
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {threatIcon(currentResult.threatLevel)}
                    <span>Detected Threats</span>
                  </div>
                  {currentResult.safeAction && (
                    <Badge variant="outline" className="text-xs font-normal">
                      <Zap className="h-3 w-3 mr-1" />
                      {currentResult.safeAction}
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                {currentResult.threats.length === 0 ? (
                  <div className="flex items-center gap-2 py-2 text-[var(--nav-low-badge-text)]">
                    <ShieldCheck className="h-4 w-4" />
                    <p className="text-sm">Path is clear — safe to proceed</p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {currentResult.threats.map((threat, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between rounded-lg border border-border bg-muted/50 px-3 py-2.5"
                      >
                        <div className="flex items-center gap-2.5 min-w-0">
                          <span className="text-sm">🔍</span>
                          <span className="text-sm font-medium truncate">
                            {threat.label}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 shrink-0 ml-3">
                          <Badge
                            variant="outline"
                            className="text-[10px] font-mono uppercase tracking-wider"
                            style={{
                              backgroundColor:
                                threat.urgency === 'high'
                                  ? 'var(--nav-high-badge-bg)'
                                  : threat.urgency === 'medium'
                                    ? 'var(--nav-medium-badge-bg)'
                                    : 'var(--nav-low-badge-bg)',
                              borderColor: 'transparent',
                              color:
                                threat.urgency === 'high'
                                  ? 'var(--nav-high-badge-text)'
                                  : threat.urgency === 'medium'
                                    ? 'var(--nav-medium-badge-text)'
                                    : 'var(--nav-low-badge-text)',
                            }}
                          >
                            {threat.urgency}
                          </Badge>
                          <Badge variant="secondary" className="text-[10px] font-mono">
                            {positionLabel(threat.position)}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* ─── Scene Summary ───────────────────────────────────────── */}
          {currentResult?.summary && (
            <div className="rounded-xl border border-border bg-muted/30 px-5 py-3">
              <p className="text-sm text-muted-foreground leading-relaxed">
                <span className="font-semibold text-foreground">Scene: </span>
                {currentResult.summary}
              </p>
            </div>
          )}

          {/* ─── Controls ────────────────────────────────────────────── */}
          <Card className="border-border">
            <CardContent className="pt-5">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {/* Start / Stop Camera */}
                {!cameraActive ? (
                  <Button
                    onClick={handleStart}
                    className="col-span-2 sm:col-span-1 gap-2"
                    size="lg"
                  >
                    <Camera className="h-5 w-5" />
                    Start Camera
                  </Button>
                ) : (
                  <Button
                    onClick={handleStop}
                    variant="destructive"
                    className="col-span-2 sm:col-span-1 gap-2"
                    size="lg"
                  >
                    <CameraOff className="h-5 w-5" />
                    Stop
                  </Button>
                )}

                {/* Manual Scan */}
                <Button
                  onClick={handleManualScan}
                  disabled={!cameraActive || isScanning}
                  variant="outline"
                  className="col-span-1 gap-2"
                  size="lg"
                >
                  <Eye className="h-5 w-5" />
                  Scan Now
                </Button>

                {/* Pause / Resume */}
                <Button
                  onClick={handlePauseResume}
                  disabled={!cameraActive}
                  variant={isPaused ? 'default' : 'outline'}
                  className="col-span-1 gap-2"
                  size="lg"
                >
                  {isPaused ? (
                    <>
                      <Play className="h-5 w-5" />
                      Resume
                    </>
                  ) : (
                    <>
                      <Pause className="h-5 w-5" />
                      Pause
                    </>
                  )}
                </Button>

                {/* Frequency Selector */}
                <div className="col-span-2 sm:col-span-1 flex items-center gap-2">
                  <div className="flex rounded-lg border border-border overflow-hidden w-full">
                    {([4000, 6000, 10000] as ScanFrequency[]).map((freq) => (
                      <button
                        key={freq}
                        onClick={() => setScanFrequency(freq)}
                        className={`flex-1 px-3 py-2.5 text-xs font-mono font-medium transition-colors ${
                          scanFrequency === freq
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted/50 text-muted-foreground hover:bg-muted'
                        }`}
                      >
                        {frequencyLabel(freq)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Test Voice */}
                <Button
                  onClick={testVoice}
                  variant="outline"
                  className="col-span-2 sm:col-span-1 gap-2"
                  size="lg"
                >
                  <Volume2 className="h-5 w-5" />
                  Test Voice
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* ─── Scan History ────────────────────────────────────────── */}
          {scanHistory.length > 1 && (
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <RotateCcw className="h-4 w-4" />
                  Recent Scans ({scanHistory.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                  {scanHistory.slice(1).map((scan, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 text-xs text-muted-foreground py-1.5 border-b border-border/50 last:border-0"
                    >
                      <Badge
                        variant="outline"
                        className={`text-[10px] font-mono ${
                          scan.threatLevel === 'danger'
                            ? 'border-[var(--nav-high-badge-text)] text-[var(--nav-high-badge-text)]'
                            : scan.threatLevel === 'caution'
                              ? 'border-[var(--nav-medium-badge-text)] text-[var(--nav-medium-badge-text)]'
                              : 'border-[var(--nav-low-badge-text)] text-[var(--nav-low-badge-text)]'
                        }`}
                      >
                        {scan.threatLevel}
                      </Badge>
                      <span className="truncate flex-1">{scan.voiceAlert}</span>
                      {scan.threats.length > 0 && (
                        <Badge variant="secondary" className="text-[10px] shrink-0">
                          {scan.threats.length} threat{scan.threats.length > 1 ? 's' : ''}
                        </Badge>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* ─── Camera Error (full page) ───────────────────────────── */}
          {cameraError && !cameraActive && (
            <Card className="border-destructive/50 bg-destructive/5">
              <CardContent className="pt-5">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="h-6 w-6 text-destructive shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-sm text-destructive mb-1">
                      Camera Access Required
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {cameraError}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </main>

        {/* ─── Footer ────────────────────────────────────────────────── */}
        <footer
          className="border-t px-4 py-3 text-center text-xs"
          style={{
            backgroundColor: 'var(--nav-footer-bg)',
            color: 'var(--nav-footer-text)',
            borderColor: 'var(--nav-footer-border)',
          }}
        >
          <p className="flex items-center justify-center gap-1.5">
            <Eye className="h-3.5 w-3.5" />
            NavAssist — AI-Powered Blind Navigation Assistant
          </p>
          <p className="mt-0.5 opacity-70">
            For accessibility assistance only. Not a replacement for mobility aids.
          </p>
        </footer>
      </div>
    </>
  );
}
