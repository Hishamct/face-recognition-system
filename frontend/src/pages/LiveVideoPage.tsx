import React, { useState, useRef } from 'react';
import {
  Video,
  Camera,
  Play,
  Square,
  Activity,
  Cpu,
  Clock,
  UserCheck,
  UserX,
  Upload,
  Film,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Sliders,
  Eye,
  Trash2,
  FileVideo,
} from 'lucide-react';
import { useWebcamStream } from '../hooks/useWebcamStream';
import { VideoCanvas } from '../components/VideoCanvas';
import { api } from '../services/api';
import type { VideoSummaryResponse } from '../types/api.types';

export const LiveVideoPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'webcam' | 'file'>('webcam');
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [showScores, setShowScores] = useState<boolean>(true);

  // Hook for webcam stream and WebSocket communication
  const {
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
  } = useWebcamStream({
    targetFps: 10,
    initialThreshold: 0.5697,
  });

  // State for Video File Ingestion tab
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileStride, setFileStride] = useState<number>(5);
  const [fileThreshold, setFileThreshold] = useState<number>(0.5697);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<VideoSummaryResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadError(null);
      setUploadResult(null);
    }
  };

  const handleProcessVideo = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setUploadError(null);
    try {
      const res = await api.uploadVideo(selectedFile, fileStride, fileThreshold);
      setUploadResult(res);
    } catch (err: any) {
      setUploadError(
        err.response?.data?.detail || err.message || 'Failed to process video file on server'
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Video className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white">
                Real-Time Video Stream & Tracking
              </h1>
              <p className="text-xs text-slate-400">
                Low-latency biometrics via WebSocket streaming & offline batch video analysis.
              </p>
            </div>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center rounded-lg bg-slate-900 border border-slate-800 p-1">
          <button
            onClick={() => setActiveTab('webcam')}
            className={`flex items-center space-x-2 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              activeTab === 'webcam'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Camera className="h-3.5 w-3.5" />
            <span>Live Webcam</span>
          </button>
          <button
            onClick={() => setActiveTab('file')}
            className={`flex items-center space-x-2 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              activeTab === 'file'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Film className="h-3.5 w-3.5" />
            <span>Video File Analyzer</span>
          </button>
        </div>
      </div>

      {/* TAB 1: LIVE WEBCAM STREAM */}
      {activeTab === 'webcam' && (
        <div className="space-y-6">
          {/* Controls Bar */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-4">
                {/* Device Selector */}
                <div className="flex items-center space-x-2">
                  <label htmlFor="camera-select" className="text-xs font-medium text-slate-400">Camera:</label>
                  <select
                    id="camera-select"
                    value={selectedDeviceId}
                    onChange={(e) => setSelectedDeviceId(e.target.value)}
                    disabled={isStreaming}
                    className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none disabled:opacity-50"
                  >
                    {devices.length === 0 ? (
                      <option value="">Default Camera</option>
                    ) : (
                      devices.map((d, i) => (
                        <option key={d.deviceId || i} value={d.deviceId}>
                          {d.label || `Camera ${i + 1}`}
                        </option>
                      ))
                    )}
                  </select>
                </div>

                {/* Target FPS Selector */}
                <div className="flex items-center space-x-2">
                  <label htmlFor="target-fps-select" className="text-xs font-medium text-slate-400">FPS Throttle:</label>
                  <select
                    id="target-fps-select"
                    value={targetFps}
                    onChange={(e) => setTargetFps(Number(e.target.value))}
                    className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
                  >
                    <option value={5}>5 FPS (Ultra-low CPU)</option>
                    <option value={10}>10 FPS (Recommended)</option>
                    <option value={15}>15 FPS (Smooth)</option>
                    <option value={20}>20 FPS (High Performance)</option>
                  </select>
                </div>

                {/* Threshold Slider */}
                <div className="flex items-center space-x-2">
                  <label htmlFor="threshold-slider" className="text-xs font-medium text-slate-400">
                    Threshold: <span className="text-slate-200 font-mono">{(confidenceThreshold * 100).toFixed(0)}%</span>
                  </label>
                  <input
                    id="threshold-slider"
                    type="range"
                    min="0.30"
                    max="0.85"
                    step="0.01"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="h-1.5 w-24 cursor-pointer accent-indigo-500"
                  />
                </div>

                {/* Landmark & Score Toggles */}
                <div className="flex items-center space-x-3 text-xs text-slate-400">
                  <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showLandmarks}
                      onChange={(e) => setShowLandmarks(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-0"
                    />
                    <span>Landmarks</span>
                  </label>
                  <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showScores}
                      onChange={(e) => setShowScores(e.target.checked)}
                      className="rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-0"
                    />
                    <span>Confidence</span>
                  </label>
                </div>
              </div>

              {/* Start / Stop Button */}
              <div>
                {!isStreaming ? (
                  <button
                    onClick={startStream}
                    disabled={telemetry.status === 'requesting'}
                    className="flex items-center space-x-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-xs font-semibold text-white shadow-sm transition-all disabled:opacity-50"
                  >
                    {telemetry.status === 'requesting' ? (
                      <RefreshCw className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 fill-current" />
                    )}
                    <span>{telemetry.status === 'requesting' ? 'Connecting...' : 'Start Webcam Stream'}</span>
                  </button>
                ) : (
                  <button
                    onClick={stopStream}
                    className="flex items-center space-x-2 rounded-lg bg-rose-600 hover:bg-rose-500 px-4 py-2 text-xs font-semibold text-white shadow-sm transition-all"
                  >
                    <Square className="h-4 w-4 fill-current" />
                    <span>Stop Stream</span>
                  </button>
                )}
              </div>
            </div>

            {/* Error Message */}
            {telemetry.errorMessage && (
              <div className="mt-3 flex items-center space-x-2 rounded-lg bg-rose-500/10 border border-rose-500/20 px-3 py-2 text-xs text-rose-400">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{telemetry.errorMessage}</span>
              </div>
            )}
          </div>

          {/* Main Grid: Stream Viewport + Telemetry & Logs */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Video Viewport */}
            <div className="lg:col-span-8 flex flex-col space-y-3">
              <div className="relative aspect-video w-full rounded-xl border border-slate-800 bg-slate-950 overflow-hidden shadow-2xl flex items-center justify-center">
                {/* Background Grid Pattern */}
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b08_1px,transparent_1px),linear-gradient(to_bottom,#1e293b08_1px,transparent_1px)] bg-[size:24px_24px]" />

                {/* HTML5 Video Element */}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full h-full object-contain ${isStreaming ? 'block' : 'hidden'}`}
                />

                {/* Overlay Canvas for Real-Time Bounding Boxes and Landmarks */}
                {isStreaming && (
                  <VideoCanvas
                    faces={faces}
                    frameDimensions={frameDimensions}
                    showLandmarks={showLandmarks}
                    showScores={showScores}
                  />
                )}

                {/* Standby State Display */}
                {!isStreaming && (
                  <div className="relative z-10 flex flex-col items-center text-center p-8 max-w-sm">
                    <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-900 border border-slate-800 text-indigo-400 mb-4 shadow-inner">
                      <Camera className="h-8 w-8 text-indigo-400" />
                    </div>
                    <h3 className="text-sm font-semibold text-white">Camera Standby</h3>
                    <p className="mt-1 text-xs text-slate-400 leading-relaxed">
                      Select your camera input above and click &quot;Start Webcam Stream&quot; to begin full-duplex WebSocket inference.
                    </p>
                  </div>
                )}

                {/* Live Floating HUD Overlays */}
                {isStreaming && (
                  <>
                    {/* Top-Left Telemetry Tag */}
                    <div className="absolute top-3 left-3 flex items-center space-x-2 rounded-md bg-slate-950/80 border border-slate-800/80 px-2.5 py-1 text-[11px] font-mono text-slate-300 backdrop-blur-md">
                      <span className="flex h-2 w-2 relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                      </span>
                      <span>LIVE</span>
                      <span className="text-slate-600">|</span>
                      <span>{telemetry.fps} FPS</span>
                      <span className="text-slate-600">|</span>
                      <span>{telemetry.latencyMs}ms RTT</span>
                    </div>

                    {/* Top-Right Target Summary */}
                    <div className="absolute top-3 right-3 flex items-center space-x-2 rounded-md bg-slate-950/80 border border-slate-800/80 px-2.5 py-1 text-[11px] font-medium text-slate-300 backdrop-blur-md">
                      <span>
                        {faces.length === 0
                          ? 'No faces detected'
                          : `${faces.length} Face${faces.length > 1 ? 's' : ''}`}
                      </span>
                      {faces.length > 0 && (
                        <span className="text-xs">
                          (<span className="text-indigo-400">{telemetry.recognizedCount} Rec</span>,{' '}
                          <span className="text-amber-400">{telemetry.unknownCount} Unk</span>)
                        </span>
                      )}
                    </div>

                    {/* Bottom-Left Frame Size Badge */}
                    <div className="absolute bottom-3 left-3 rounded bg-slate-950/70 border border-slate-800/60 px-2 py-0.5 text-[10px] font-mono text-slate-400">
                      {frameDimensions.width}x{frameDimensions.height} px
                    </div>
                  </>
                )}
              </div>

              {/* Real-Time Metrics Strip */}
              <div className="grid grid-cols-4 gap-3">
                <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
                  <div className="flex items-center space-x-2 text-slate-400 text-[11px]">
                    <Activity className="h-3.5 w-3.5 text-indigo-400" />
                    <span>Realized FPS</span>
                  </div>
                  <div className="mt-1 text-base font-bold text-white font-mono">
                    {telemetry.fps}{' '}
                    <span className="text-[10px] font-normal text-slate-400">/ {targetFps}</span>
                  </div>
                </div>

                <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
                  <div className="flex items-center space-x-2 text-slate-400 text-[11px]">
                    <Clock className="h-3.5 w-3.5 text-cyan-400" />
                    <span>Roundtrip RTT</span>
                  </div>
                  <div className="mt-1 text-base font-bold text-white font-mono">
                    {telemetry.latencyMs}{' '}
                    <span className="text-[10px] font-normal text-slate-400">ms</span>
                  </div>
                </div>

                <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
                  <div className="flex items-center space-x-2 text-slate-400 text-[11px]">
                    <Cpu className="h-3.5 w-3.5 text-emerald-400" />
                    <span>Inference</span>
                  </div>
                  <div className="mt-1 text-base font-bold text-white font-mono">
                    {Math.round(telemetry.inferenceTimeMs)}{' '}
                    <span className="text-[10px] font-normal text-slate-400">ms</span>
                  </div>
                </div>

                <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
                  <div className="flex items-center space-x-2 text-slate-400 text-[11px]">
                    <UserCheck className="h-3.5 w-3.5 text-indigo-400" />
                    <span>Identified</span>
                  </div>
                  <div className="mt-1 text-base font-bold text-indigo-400 font-mono">
                    {telemetry.recognizedCount}{' '}
                    <span className="text-[10px] font-normal text-slate-400">
                      / {telemetry.faceCount}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Sidebar: Real-Time Event Log */}
            <div className="lg:col-span-4 flex flex-col space-y-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur-sm flex flex-col flex-1 h-[480px]">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center space-x-2">
                    <Activity className="h-4 w-4 text-indigo-400" />
                    <h3 className="text-xs font-semibold text-white">Live Recognition Log</h3>
                  </div>
                  {eventLog.length > 0 && (
                    <button
                      onClick={clearLog}
                      className="text-slate-500 hover:text-slate-300 transition-colors"
                      title="Clear log"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>

                {/* Log Feed */}
                <div className="mt-3 flex-1 overflow-y-auto space-y-2 pr-1 text-xs">
                  {eventLog.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500 text-center p-4">
                      <Clock className="h-6 w-6 mb-2 opacity-40" />
                      <p className="text-xs">No recognition events yet.</p>
                      <p className="text-[11px] text-slate-600 mt-0.5">
                        Events appear here when faces enter the camera field of view.
                      </p>
                    </div>
                  ) : (
                    eventLog.map((event) => (
                      <div
                        key={event.id}
                        className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 p-2.5 transition-colors"
                      >
                        <div className="flex items-center space-x-2.5 min-w-0">
                          <div
                            className={`flex h-7 w-7 items-center justify-center rounded-lg shrink-0 ${
                              event.isKnown
                                ? 'bg-indigo-500/10 border border-indigo-500/20 text-indigo-400'
                                : 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
                            }`}
                          >
                            {event.isKnown ? (
                              <UserCheck className="h-3.5 w-3.5" />
                            ) : (
                              <UserX className="h-3.5 w-3.5" />
                            )}
                          </div>
                          <div className="min-w-0">
                            <p className="font-semibold text-slate-200 truncate">{event.identity}</p>
                            <p className="text-[10px] text-slate-500 font-mono">{event.timestamp}</p>
                          </div>
                        </div>

                        <div className="text-right shrink-0">
                          <span
                            className={`rounded px-1.5 py-0.5 text-[10px] font-mono font-medium ${
                              event.isKnown
                                ? 'bg-indigo-500/20 text-indigo-300'
                                : 'bg-amber-500/20 text-amber-300'
                            }`}
                          >
                            {(event.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: VIDEO FILE ANALYZER */}
      {activeTab === 'file' && (
        <div className="space-y-6">
          {/* File Upload & Ingestion Settings */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur-sm space-y-6">
            <div>
              <h2 className="text-sm font-semibold text-white">Batch Video File Analysis</h2>
              <p className="mt-1 text-xs text-slate-400">
                Upload local video recordings (.mp4, .mov, .avi, .webm) to perform automated frame-strided face recognition and generate an activity timeline.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Dropzone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="md:col-span-2 flex flex-col items-center justify-center border-2 border-dashed border-slate-700 hover:border-indigo-500/50 rounded-xl p-8 cursor-pointer bg-slate-950/50 transition-colors text-center"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <FileVideo className="h-10 w-10 text-indigo-400 mb-3" />
                <p className="text-xs font-semibold text-slate-200">
                  {selectedFile ? selectedFile.name : 'Click to select or drag and drop video file'}
                </p>
                <p className="mt-1 text-[11px] text-slate-500">
                  {selectedFile
                    ? `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`
                    : 'Supports MP4, MOV, AVI, WEBM up to 100 MB'}
                </p>
              </div>

              {/* Ingestion Parameters */}
              <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                <h3 className="text-xs font-semibold text-slate-300 flex items-center space-x-1.5">
                  <Sliders className="h-3.5 w-3.5 text-indigo-400" />
                  <span>Sampling Parameters</span>
                </h3>

                <div>
                  <label htmlFor="file-stride-select" className="text-[11px] font-medium text-slate-400 block mb-1">
                    Frame Stride (Sample every N frames):
                  </label>
                  <select
                    id="file-stride-select"
                    value={fileStride}
                    onChange={(e) => setFileStride(Number(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
                  >
                    <option value={1}>Every 1st frame (100% full scan, slower)</option>
                    <option value={3}>Every 3rd frame (Thorough scan)</option>
                    <option value={5}>Every 5th frame (Balanced, default)</option>
                    <option value={10}>Every 10th frame (Fast security audit)</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="file-threshold-slider" className="text-[11px] font-medium text-slate-400 block mb-1">
                    Confidence Threshold: {(fileThreshold * 100).toFixed(0)}%
                  </label>
                  <input
                    id="file-threshold-slider"
                    type="range"
                    min="0.30"
                    max="0.85"
                    step="0.01"
                    value={fileThreshold}
                    onChange={(e) => setFileThreshold(parseFloat(e.target.value))}
                    className="w-full h-1.5 cursor-pointer accent-indigo-500"
                  />
                </div>

                <button
                  onClick={handleProcessVideo}
                  disabled={!selectedFile || isUploading}
                  className="w-full mt-2 flex items-center justify-center space-x-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-xs font-semibold text-white shadow-sm transition-all disabled:opacity-50"
                >
                  {isUploading ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      <span>Processing Video...</span>
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      <span>Analyze Video</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Upload Error */}
            {uploadError && (
              <div className="flex items-center space-x-2 rounded-lg bg-rose-500/10 border border-rose-500/20 p-3 text-xs text-rose-400">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{uploadError}</span>
              </div>
            )}
          </div>

          {/* Analysis Results Display */}
          {uploadResult && (
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur-sm space-y-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                  <h3 className="text-sm font-semibold text-white">Video Analysis Summary</h3>
                </div>
                <span className="text-xs text-slate-400 font-mono">{uploadResult.filename}</span>
              </div>

              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                  <p className="text-[11px] text-slate-400">Frames Processed</p>
                  <p className="mt-1 text-lg font-bold text-white font-mono">
                    {uploadResult.total_frames_processed}
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                  <p className="text-[11px] text-slate-400">Duration</p>
                  <p className="mt-1 text-lg font-bold text-white font-mono">
                    {uploadResult.duration_seconds.toFixed(1)}s
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                  <p className="text-[11px] text-slate-400">Average Throughput</p>
                  <p className="mt-1 text-lg font-bold text-indigo-400 font-mono">
                    {uploadResult.average_fps.toFixed(1)} FPS
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                  <p className="text-[11px] text-slate-400">Mean Latency</p>
                  <p className="mt-1 text-lg font-bold text-cyan-400 font-mono">
                    {uploadResult.average_latency_ms.toFixed(1)} ms
                  </p>
                </div>
              </div>

              {/* Unique Identities Pill List */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-slate-300">Identities Detected in Footage</h4>
                <div className="flex flex-wrap gap-2">
                  {uploadResult.unique_identities_detected.length === 0 ? (
                    <span className="text-xs text-slate-500 italic">No registered identities detected.</span>
                  ) : (
                    uploadResult.unique_identities_detected.map((name, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center space-x-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 px-3 py-1 text-xs font-medium text-indigo-300"
                      >
                        <UserCheck className="h-3.5 w-3.5" />
                        <span>{name}</span>
                      </span>
                    ))
                  )}
                </div>
              </div>

              {/* Timeline Sample Table */}
              {uploadResult.timeline_sample && uploadResult.timeline_sample.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-xs font-semibold text-slate-300">Detection Timeline Sample</h4>
                  <div className="overflow-x-auto rounded-lg border border-slate-800">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="bg-slate-950 text-slate-400 text-[11px] uppercase tracking-wider">
                        <tr>
                          <th className="px-4 py-2.5">Frame #</th>
                          <th className="px-4 py-2.5">Timestamp</th>
                          <th className="px-4 py-2.5">Face Count</th>
                          <th className="px-4 py-2.5">Detected Identities</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800 bg-slate-900/40 font-mono text-[11px]">
                        {uploadResult.timeline_sample.map((item, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                            <td className="px-4 py-2 text-slate-400">{item.frame}</td>
                            <td className="px-4 py-2 text-slate-300">
                              {item.timestamp_seconds.toFixed(2)}s
                            </td>
                            <td className="px-4 py-2">{item.face_count}</td>
                            <td className="px-4 py-2 text-indigo-300 font-sans">
                              {item.identities.length > 0
                                ? item.identities.join(', ')
                                : 'None / Unknown'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LiveVideoPage;
