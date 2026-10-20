import React, { useEffect, useRef } from 'react';
import type { FaceRecognitionItem } from '../types/api.types';

interface VideoCanvasProps {
  faces: FaceRecognitionItem[];
  frameDimensions: { width: number; height: number };
  showLandmarks?: boolean;
  showScores?: boolean;
  className?: string;
}

export const VideoCanvas: React.FC<VideoCanvasProps> = ({
  faces,
  frameDimensions,
  showLandmarks = true,
  showScores = true,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Synchronize canvas internal buffer with display size
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Use display size multiplied by DPR for high-DPI crisp rendering
    const displayW = Math.round(rect.width);
    const displayH = Math.round(rect.height);

    if (canvas.width !== displayW * dpr || canvas.height !== displayH * dpr) {
      canvas.width = displayW * dpr;
      canvas.height = displayH * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayW, displayH);

    if (!faces || faces.length === 0 || !frameDimensions.width || !frameDimensions.height) {
      ctx.restore();
      return;
    }

    // Scale factors from backend frame coordinates to display canvas coordinates
    const scaleX = displayW / frameDimensions.width;
    const scaleY = displayH / frameDimensions.height;

    for (const face of faces) {
      const bbox = face.bbox;
      if (!bbox) continue;

      const rawX = bbox.x1 ?? bbox.x ?? 0;
      const rawY = bbox.y1 ?? bbox.y ?? 0;
      const rawW = bbox.width || (bbox.x2 ? bbox.x2 - rawX : 0);
      const rawH = bbox.height || (bbox.y2 ? bbox.y2 - rawY : 0);

      const x = rawX * scaleX;
      const y = rawY * scaleY;
      const w = rawW * scaleX;
      const h = rawH * scaleY;

      const isRecognized = face.status === 'RECOGNIZED' || face.is_known;
      const mainColor = isRecognized ? '#6366F1' : '#F59E0B'; // Indigo for recognized, Amber for unknown
      const cornerColor = isRecognized ? '#818CF8' : '#FBBF24';
      const bgColor = isRecognized ? 'rgba(99, 102, 241, 0.08)' : 'rgba(245, 158, 11, 0.08)';

      // 1. Semi-transparent bounding box fill
      ctx.fillStyle = bgColor;
      ctx.fillRect(x, y, w, h);

      // 2. Base bounding box outline
      ctx.strokeStyle = mainColor;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x, y, w, h);

      // 3. Technical HUD Corner Brackets
      const cornerLen = Math.max(10, Math.min(22, Math.min(w, h) * 0.2));
      ctx.strokeStyle = cornerColor;
      ctx.lineWidth = 3;
      ctx.lineCap = 'square';

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x, y + cornerLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLen, y);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerLen);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerLen, y + h);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerLen);
      ctx.stroke();

      // 4. Biometric 5-Point Landmarks
      if (showLandmarks && face.landmarks) {
        const lm = face.landmarks;
        const points = [
          lm.right_eye,
          lm.left_eye,
          lm.nose_tip,
          lm.mouth_right || lm.right_mouth_corner,
          lm.mouth_left || lm.left_mouth_corner,
        ].filter(Boolean);

        for (const pt of points) {
          if (!pt) continue;
          const px = pt.x * scaleX;
          const py = pt.y * scaleY;

          // Outer subtle aura
          ctx.beginPath();
          ctx.arc(px, py, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
          ctx.fill();

          // Core landmark dot
          ctx.beginPath();
          ctx.arc(px, py, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = '#38BDF8'; // Sky 400
          ctx.fill();
        }
      }

      // 5. Identification Header Pill
      const name = face.matched_name || face.identity || 'Unknown Face';
      const confScore = Math.round((face.confidence ?? 0) * 100);
      const labelText = showScores ? `${name} (${confScore}%)` : name;

      ctx.font = '600 11px system-ui, -apple-system, sans-serif';
      const textMetrics = ctx.measureText(labelText);
      const pillW = textMetrics.width + 20;
      const pillH = 22;
      const pillX = x;
      const pillY = Math.max(4, y - pillH - 4);

      // Pill background
      ctx.fillStyle = isRecognized ? '#1E1B4B' : '#451A03';
      ctx.strokeStyle = isRecognized ? 'rgba(99, 102, 241, 0.45)' : 'rgba(245, 158, 11, 0.45)';
      ctx.lineWidth = 1;

      // Rounded rectangle for pill
      ctx.beginPath();
      const r = 4;
      ctx.moveTo(pillX + r, pillY);
      ctx.lineTo(pillX + pillW - r, pillY);
      ctx.arcTo(pillX + pillW, pillY, pillX + pillW, pillY + r, r);
      ctx.lineTo(pillX + pillW, pillY + pillH - r);
      ctx.arcTo(pillX + pillW, pillY + pillH, pillX + pillW - r, pillY + pillH, r);
      ctx.lineTo(pillX + r, pillY + pillH);
      ctx.arcTo(pillX, pillY + pillH, pillX, pillY + pillH - r, r);
      ctx.lineTo(pillX, pillY + r);
      ctx.arcTo(pillX, pillY, pillX + r, pillY, r);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Status indicator dot
      ctx.beginPath();
      ctx.arc(pillX + 8, pillY + pillH / 2, 3, 0, Math.PI * 2);
      ctx.fillStyle = isRecognized ? '#10B981' : '#F59E0B'; // Green or Amber dot
      ctx.fill();

      // Pill text
      ctx.fillStyle = isRecognized ? '#F8FAFC' : '#FEF3C7';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, pillX + 16, pillY + pillH / 2);
    }

    ctx.restore();
  }, [faces, frameDimensions, showLandmarks, showScores]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none w-full h-full ${className}`}
    />
  );
};

export default VideoCanvas;
