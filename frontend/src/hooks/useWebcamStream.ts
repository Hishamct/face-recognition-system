import { useState, useEffect, useRef, useCallback } from 'react';
import type { FaceRecognitionItem, VideoStreamFrameResponse } from '../types/api.types';
import { api } from '../services/api';

export interface StreamTelemetry {
  fps: number;
  latencyMs: number;
  inferenceTimeMs: number;
  faceCount: number;
  recognizedCount: number;
  unknownCount: number;
  status: 'idle' | 'requesting' | 'connected' | 'streaming' | 'error' | 'disconnected';
  errorMessage?: string;
}

export interface DetectionEvent {
  id: string;
  timestamp: string;
  identity: string;
  confidence: number;
  isKnown: boolean;
  status: string;
}

export interface UseWebcamStreamOptions {
  targetFps?: number;
  initialThreshold?: number;
  maxLogItems?: number;
}

export interface UseWebcamStreamReturn {
  videoRef: React.RefObject<HTMLVideoElement>;
  devices: MediaDeviceInfo[];
  selectedDeviceId: string;
  setSelectedDeviceId: (id: string) => void;
  telemetry: StreamTelemetry;
  faces: FaceRecognitionItem[];
  frameDimensions: { width: number; height: number };
  eventLog: DetectionEvent[];
  isStreaming: boolean;
  confidenceThreshold: number;
  setConfidenceThreshold: (val: number) => void;
  targetFps: number;
  setTargetFps: (fps: number) => void;
  startStream: () => Promise<void>;
  stopStream: () => void;
  clearLog: () => void;
}

export function useWebcamStream(options: UseWebcamStreamOptions = {}): UseWebcamStreamReturn {
  const {
    targetFps: initialTargetFps = 10,
    initialThreshold = 0.5697,
    maxLogItems = 40,
  } = options;

  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const isProcessingRef = useRef<boolean>(false);
  const sendTimeRef = useRef<number>(0);
  const frameIntervalRef = useRef<number | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fpsHistoryRef = useRef<number[]>([]);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [targetFps, setTargetFps] = useState<number>(initialTargetFps);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(initialThreshold);
  const [faces, setFaces] = useState<FaceRecognitionItem[]>([]);
  const [frameDimensions, setFrameDimensions] = useState<{ width: number; height: number }>({ width: 640, height: 480 });
  const [eventLog, setEventLog] = useState<DetectionEvent[]>([]);

  const [telemetry, setTelemetry] = useState<StreamTelemetry>({
    fps: 0,
    latencyMs: 0,
    inferenceTimeMs: 0,
    faceCount: 0,
    recognizedCount: 0,
    unknownCount: 0,
    status: 'idle',
  });

  // Enumerate camera devices
  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter((d) => d.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    } catch {
      // Device enumeration not allowed or failed
    }
  }, [selectedDeviceId]);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  // Stop camera and WebSocket
  const stopStream = useCallback(() => {
    if (frameIntervalRef.current) {
      window.clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      wsRef.current.close();
      wsRef.current = null;
    }

    isProcessingRef.current = false;
    fpsHistoryRef.current = [];
    setIsStreaming(false);
    setFaces([]);
    setTelemetry((prev) => ({
      ...prev,
      status: 'idle',
      fps: 0,
      faceCount: 0,
      recognizedCount: 0,
      unknownCount: 0,
    }));
  }, []);

  // Frame sender function
  const sendCurrentFrame = useCallback(() => {
    const video = videoRef.current;
    const ws = wsRef.current;

    if (
      !video ||
      !ws ||
      ws.readyState !== WebSocket.OPEN ||
      isProcessingRef.current ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.paused ||
      video.ended
    ) {
      return;
    }

    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
    }

    const canvas = offscreenCanvasRef.current;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    // Multi-resolution downscaling for CPU inference efficiency
    let targetW = width;
    let targetH = height;
    const maxDim = 640;
    if (Math.max(width, height) > maxDim) {
      const scale = maxDim / Math.max(width, height);
      targetW = Math.round(width * scale);
      targetH = Math.round(height * scale);
    }

    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, targetW, targetH);

    isProcessingRef.current = true;
    sendTimeRef.current = performance.now();

    const dataUrl = canvas.toDataURL('image/jpeg', 0.72);
    ws.send(
      JSON.stringify({
        frame: dataUrl,
        threshold: confidenceThreshold,
      })
    );
  }, [confidenceThreshold]);

  // Start webcam and WebSocket connection
  const startStream = useCallback(async () => {
    stopStream();

    setTelemetry((prev) => ({ ...prev, status: 'requesting', errorMessage: undefined }));

    try {
      // 1. Acquire Webcam Media Stream
      const constraints: MediaStreamConstraints = {
        video: selectedDeviceId
          ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      mediaStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      await refreshDevices();

      // 2. Open WebSocket
      setTelemetry((prev) => ({ ...prev, status: 'connected' }));
      const wsUrl = api.getVideoWebSocketUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsStreaming(true);
        setTelemetry((prev) => ({ ...prev, status: 'streaming' }));

        // Start capture loop
        const intervalMs = Math.max(30, Math.floor(1000 / targetFps));
        frameIntervalRef.current = window.setInterval(sendCurrentFrame, intervalMs);
      };

      ws.onmessage = (event: MessageEvent) => {
        const roundtripLatency = Math.round(performance.now() - sendTimeRef.current);
        isProcessingRef.current = false;

        try {
          const data: VideoStreamFrameResponse = JSON.parse(event.data);

          if (data.success) {
            const detectedFaces = data.faces || [];
            setFaces(detectedFaces);
            if (data.image_width && data.image_height) {
              setFrameDimensions({ width: data.image_width, height: data.image_height });
            }

            // Compute FPS
            const currentFps = roundtripLatency > 0 ? 1000 / roundtripLatency : data.fps;
            fpsHistoryRef.current.push(currentFps);
            if (fpsHistoryRef.current.length > 8) {
              fpsHistoryRef.current.shift();
            }
            const avgFps = Math.round(
              fpsHistoryRef.current.reduce((a, b) => a + b, 0) / fpsHistoryRef.current.length
            );

            const recognized = detectedFaces.filter(
              (f) => f.status === 'RECOGNIZED' || f.is_known
            ).length;
            const unknown = detectedFaces.length - recognized;

            setTelemetry({
              fps: Math.min(avgFps, targetFps),
              latencyMs: roundtripLatency,
              inferenceTimeMs: data.inference_time_ms,
              faceCount: detectedFaces.length,
              recognizedCount: recognized,
              unknownCount: unknown,
              status: 'streaming',
            });

            // Log recognized events
            if (detectedFaces.length > 0) {
              const nowStr = new Date().toLocaleTimeString();
              const newEvents: DetectionEvent[] = detectedFaces.map((f, idx) => ({
                id: `${Date.now()}-${idx}`,
                timestamp: nowStr,
                identity: f.matched_name || f.identity || 'Unknown Face',
                confidence: f.confidence,
                isKnown: f.is_known ?? f.status === 'RECOGNIZED',
                status: f.status || (f.is_known ? 'RECOGNIZED' : 'UNKNOWN'),
              }));

              setEventLog((prev) => {
                const combined = [...newEvents, ...prev];
                return combined.slice(0, maxLogItems);
              });
            }
          } else {
            setTelemetry((prev) => ({
              ...prev,
              errorMessage: data.error || 'Server error processing frame',
            }));
          }
        } catch {
          // Parsing error
        }
      };

      ws.onerror = () => {
        isProcessingRef.current = false;
        setTelemetry((prev) => ({
          ...prev,
          status: 'error',
          errorMessage: 'WebSocket connection to face recognition backend failed',
        }));
      };

      ws.onclose = () => {
        isProcessingRef.current = false;
        setIsStreaming(false);
        setTelemetry((prev) => ({
          ...prev,
          status: 'disconnected',
        }));
      };
    } catch (err: any) {
      stopStream();
      setTelemetry((prev) => ({
        ...prev,
        status: 'error',
        errorMessage: err.message || 'Failed to initialize camera or media stream',
      }));
    }
  }, [selectedDeviceId, targetFps, sendCurrentFrame, stopStream, refreshDevices, maxLogItems]);

  // Adjust interval when targetFps changes dynamically
  useEffect(() => {
    if (isStreaming && wsRef.current?.readyState === WebSocket.OPEN) {
      if (frameIntervalRef.current) {
        window.clearInterval(frameIntervalRef.current);
      }
      const intervalMs = Math.max(30, Math.floor(1000 / targetFps));
      frameIntervalRef.current = window.setInterval(sendCurrentFrame, intervalMs);
    }
  }, [targetFps, isStreaming, sendCurrentFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, [stopStream]);

  const clearLog = useCallback(() => {
    setEventLog([]);
  }, []);

  return {
    videoRef,
    devices,
    selectedDeviceId,
    setSelectedDeviceId,
    telemetry,
    faces,
    frameDimensions,
    eventLog,
    isStreaming,
    confidenceThreshold,
    setConfidenceThreshold,
    targetFps,
    setTargetFps,
    startStream,
    stopStream,
    clearLog,
  };
}
