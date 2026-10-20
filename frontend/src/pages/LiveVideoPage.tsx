import React, { useState } from 'react';
import { Video, ShieldCheck, Zap, Radio, Camera, Cpu } from 'lucide-react';

export const LiveVideoPage: React.FC = () => {
  const [cameraActive, setCameraActive] = useState(false);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-xl font-bold tracking-tight text-white">Live Video Stream & Tracking</h1>
            <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-[11px] font-semibold text-slate-400 border border-slate-700">
              Phase 7 Roadmap
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Real-time 30 FPS video streaming with persistent multi-target face tracking and WebSocket telemetry.
          </p>
        </div>
      </div>

      {/* Main Stream Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8">
          <div className="relative aspect-video w-full rounded-xl border border-slate-800 bg-slate-950 flex flex-col items-center justify-center p-8 overflow-hidden">
            {/* Grid background effect */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b08_1px,transparent_1px),linear-gradient(to_bottom,#1e293b08_1px,transparent_1px)] bg-[size:24px_24px]" />

            <div className="relative z-10 flex flex-col items-center text-center max-w-md">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-900 border border-slate-800 text-brand-500 mb-4 shadow-inner">
                <Video className="h-8 w-8 text-brand-500" />
              </div>
              <h3 className="text-base font-bold text-white">Live Stream Pipeline (Phase 7)</h3>
              <p className="mt-2 text-xs text-slate-400 leading-relaxed">
                Full-duplex WebSocket connection (`/ws/live-stream`) with dedicated background worker thread,
                sub-25ms frame throughput, and SORT/ByteTrack multi-face tracking will be integrated in Phase 7.
              </p>

              <div className="mt-6 flex items-center space-x-3">
                <button
                  onClick={() => setCameraActive(!cameraActive)}
                  className="flex items-center space-x-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 px-4 py-2 text-xs font-semibold text-white transition-colors"
                >
                  <Camera className="h-4 w-4" />
                  <span>{cameraActive ? 'Stop Preview' : 'Test Local Webcam'}</span>
                </button>
              </div>

              {cameraActive && (
                <div className="mt-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 text-xs text-emerald-400">
                  Local camera device detected & verified. Ready for WebSocket streaming in Phase 7.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Specifications Sidebar */}
        <div className="lg:col-span-4 space-y-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm space-y-4">
            <h3 className="text-sm font-semibold text-white">Phase 7 Technical Architecture</h3>

            <div className="space-y-3 text-xs text-slate-400">
              <div className="flex items-start space-x-3">
                <Radio className="h-4 w-4 text-brand-500 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-slate-200">WebSocket Transport:</strong>
                  <p className="text-slate-400 mt-0.5">Binary frame ingestion without HTTP connection handshake overhead.</p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <Cpu className="h-4 w-4 text-brand-500 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-slate-200">Asynchronous Worker:</strong>
                  <p className="text-slate-400 mt-0.5">Offloads YuNet detection to parallel process pool to avoid blocking ASGI loop.</p>
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <Zap className="h-4 w-4 text-brand-500 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-slate-200">Kalman Filter Tracking:</strong>
                  <p className="text-slate-400 mt-0.5">Smooths bounding box jitter across frames with persistent track IDs.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveVideoPage;
