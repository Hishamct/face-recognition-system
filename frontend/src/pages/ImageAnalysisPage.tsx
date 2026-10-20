import React, { useState } from 'react';
import {
  ScanFace,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Sliders,
  ChevronRight,
  Info,
  Loader2,
} from 'lucide-react';
import api from '../services/api';
import FaceCanvas from '../components/FaceCanvas';
import ImageUploader from '../components/ImageUploader';
import type { FaceRecognitionItem, RecognitionResponse } from '../types/api.types';
import { useAnalysis } from '../context/AnalysisContext';

export const ImageAnalysisPage: React.FC = () => {
  const {
    selectedFile,
    previewUrl,
    result,
    selectedFaceIdx,
    similarityThreshold,
    loading,
    error,
    setSelectedFile,
    setPreviewUrl,
    setResult,
    setSelectedFaceIdx,
    setSimilarityThreshold,
    setLoading,
    setError,
    clearAnalysis,
  } = useAnalysis();

  const handleImageSelected = async (file: File) => {
    setSelectedFile(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setError(null);
    setResult(null);
    setSelectedFaceIdx(0);

    await runRecognition(file, similarityThreshold);
  };

  const handleClear = () => {
    clearAnalysis();
  };

  const runRecognition = async (file: File, threshold: number) => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.recognizeFaces(file, threshold, 5);
      setResult(data);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to analyze image';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setSimilarityThreshold(val);
    if (selectedFile) {
      runRecognition(selectedFile, val);
    }
  };

  const currentFace: FaceRecognitionItem | undefined = result?.faces?.[selectedFaceIdx];

  // Defensive field resolution
  const matchedName = currentFace
    ? currentFace.identity || currentFace.matched_name || 'Unknown'
    : 'Unknown';

  const isKnown = currentFace
    ? currentFace.status === 'RECOGNIZED' ||
      currentFace.is_known === true ||
      (currentFace.identity && currentFace.identity.toUpperCase() !== 'UNKNOWN')
    : false;

  const topCandidates = currentFace
    ? currentFace.top_candidates || currentFace.candidates || []
    : [];

  const totalFacesDetected = result
    ? result.face_count ?? result.total_faces ?? result.faces?.length ?? 0
    : 0;

  const latency = result
    ? result.latency_ms ?? result.inference_time_ms ?? 0
    : 0;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">Face Recognition & Analysis</h1>
          <p className="mt-1 text-xs text-slate-400">
            Real-time face detection, alignment, SFace 128-D embedding, and calibrated open-set identification.
          </p>
        </div>

        {/* Operating Threshold Control */}
        <div className="flex items-center space-x-3 rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-2">
          <Sliders className="h-4 w-4 text-brand-500" />
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Threshold (τ):</span>
            <input
              type="range"
              min="0.30"
              max="0.90"
              step="0.01"
              value={similarityThreshold}
              onChange={handleThresholdChange}
              className="h-1.5 w-24 cursor-pointer accent-brand-500 rounded-lg bg-slate-700"
            />
            <span className="font-mono text-xs font-semibold text-white">
              {similarityThreshold.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Canvas Viewer */}
        <div className="lg:col-span-7 flex flex-col space-y-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 backdrop-blur-sm">
            <ImageUploader
              onImageSelected={handleImageSelected}
              onClear={handleClear}
              selectedFile={selectedFile}
              loading={loading}
            />

            {error && (
              <div className="mt-4 flex items-center space-x-2 rounded-lg bg-rose-500/10 border border-rose-500/20 px-3 py-2 text-xs text-rose-400">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {previewUrl && (
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 backdrop-blur-sm relative">
              {loading && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center rounded-xl bg-slate-950/75 backdrop-blur-sm">
                  <Loader2 className="h-8 w-8 animate-spin text-brand-500" />
                  <p className="mt-2 text-xs font-medium text-slate-300">Extracting 128-D Biometric Embedding...</p>
                </div>
              )}

              <FaceCanvas
                imageSrc={previewUrl}
                faces={result?.faces ?? []}
                selectedFaceIndex={selectedFaceIdx}
                onSelectFace={(idx) => setSelectedFaceIdx(idx)}
              />

              {result && (
                <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-white">{totalFacesDetected}</span>
                    <span>{totalFacesDetected === 1 ? 'face detected' : 'faces detected'}</span>
                  </div>
                  <div className="flex items-center space-x-1.5 text-slate-400 font-mono">
                    <Clock className="h-3.5 w-3.5 text-slate-500" />
                    <span>Inference: {typeof latency === 'number' ? latency.toFixed(1) : latency} ms</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Biometric Inspection Panel */}
        <div className="lg:col-span-5 flex flex-col space-y-4">
          {currentFace ? (
            <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm space-y-5">
              {/* Decision Header */}
              <div className="flex items-start justify-between border-b border-slate-800/80 pb-4">
                <div className="flex items-center space-x-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-800 border border-slate-700 text-base font-bold text-slate-200">
                    {matchedName.slice(0, 2).toUpperCase()}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <h2 className="text-base font-bold text-white">{matchedName}</h2>
                      {isKnown ? (
                        <span className="inline-flex items-center rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 className="mr-1 h-3 w-3" />
                          VERIFIED
                        </span>
                      ) : (
                        <span className="inline-flex items-center rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] font-medium text-amber-400 border border-amber-500/20">
                          <AlertTriangle className="mr-1 h-3 w-3" />
                          UNKNOWN
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-400">
                      Face #{currentFace.face_id ?? (selectedFaceIdx + 1)} · Threshold: τ = {currentFace.threshold_used}
                    </p>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-xl font-bold font-mono text-white">
                    {(currentFace.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500">Confidence</div>
                </div>
              </div>

              {/* Top Candidates Ranking */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                  Top Gallery Matches
                </h3>
                {topCandidates.length > 0 ? (
                  <div className="space-y-2.5">
                    {topCandidates.map((cand, idx) => {
                      const candName = cand.identity || cand.identity_name || 'Unknown';
                      const candScore = cand.confidence ?? cand.similarity ?? 0;
                      const pct = (candScore * 100).toFixed(1);
                      const isTop = idx === 0 && isKnown;
                      return (
                        <div
                          key={`${candName}-${idx}`}
                          className="rounded-lg bg-slate-800/50 p-2.5 border border-slate-800"
                        >
                          <div className="flex items-center justify-between text-xs mb-1.5">
                            <span className={`font-medium ${isTop ? 'text-white' : 'text-slate-300'}`}>
                              {candName}
                            </span>
                            <span className="font-mono text-slate-400">{pct}%</span>
                          </div>
                          <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isTop ? 'bg-indigo-500' : 'bg-slate-600'
                              }`}
                              style={{ width: `${Math.min(100, Math.max(0, candScore * 100))}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">No alternate candidate scores recorded.</p>
                )}
              </div>

              {/* Bounding Box & Landmark Coordinates */}
              {currentFace.bbox && (
                <div className="rounded-lg bg-slate-950/60 p-3 border border-slate-800/80 text-xs font-mono text-slate-400 space-y-1">
                  <div className="flex justify-between">
                    <span>Bounding Box:</span>
                    <span className="text-slate-200">
                      [{currentFace.bbox.x1 ?? (currentFace.bbox as any).x ?? 0}, {currentFace.bbox.y1 ?? (currentFace.bbox as any).y ?? 0}, {currentFace.bbox.width}x{currentFace.bbox.height}]
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Landmark Points:</span>
                    <span className="text-slate-200">5 Aligned Anchor Points</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-900/30 p-12 text-center">
              <ScanFace className="h-12 w-12 text-slate-600 mb-3" />
              <h3 className="text-sm font-medium text-slate-300">No Face Selected</h3>
              <p className="mt-1 text-xs text-slate-500 max-w-xs">
                Upload or drop an image on the left to extract biometric vectors and view identity matching results.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageAnalysisPage;
