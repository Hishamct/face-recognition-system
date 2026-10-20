import React, { useEffect, useRef } from 'react';
import type { FaceRecognitionItem } from '../types/api.types';

interface FaceCanvasProps {
  imageSrc: string;
  faces: FaceRecognitionItem[];
  selectedFaceIndex?: number;
  onSelectFace?: (index: number) => void;
}

export const FaceCanvas: React.FC<FaceCanvasProps> = ({
  imageSrc,
  faces,
  selectedFaceIndex = 0,
  onSelectFace,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageSrc) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.src = imageSrc;
    img.onload = () => {
      // Set canvas dimensions to match image natural resolution
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // Draw base photo
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Dynamic scale factor based on image resolution so lines/text stay crisp on all resolutions
      const resScale = Math.max(1, Math.min(canvas.width, canvas.height) / 700);

      // Draw each face overlay
      faces.forEach((face, idx) => {
        const isSelected = idx === selectedFaceIndex;

        // Resolve coordinates (supports x1/y1 or x/y)
        const x = face.bbox.x1 ?? (face.bbox as any).x ?? 0;
        const y = face.bbox.y1 ?? (face.bbox as any).y ?? 0;
        const width = face.bbox.width ?? (face.bbox.x2 !== undefined ? face.bbox.x2 - x : 100);
        const height = face.bbox.height ?? (face.bbox.y2 !== undefined ? face.bbox.y2 - y : 100);

        // Colors (refined, non-neon)
        const isKnown =
          face.status === 'RECOGNIZED' ||
          face.is_known === true ||
          (Boolean(face.identity) && face.identity?.toUpperCase() !== 'UNKNOWN');

        const primaryColor = isKnown ? '#6366F1' : '#F59E0B'; // Indigo for known, Amber for unknown
        const fillColor = isKnown ? 'rgba(99, 102, 241, 0.12)' : 'rgba(245, 158, 11, 0.12)';

        // 1. Semi-transparent bounding box fill
        ctx.fillStyle = fillColor;
        ctx.fillRect(x, y, width, height);

        // 2. Bounding box border (scaled to resolution)
        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = Math.max(2, Math.round(resScale * (isSelected ? 3.5 : 2.5)));
        ctx.strokeRect(x, y, width, height);

        // 3. Technical corner brackets
        const bracketLen = Math.min(width, height) * 0.18;
        ctx.lineWidth = Math.max(3, Math.round(resScale * (isSelected ? 5 : 3.5)));
        ctx.strokeStyle = '#FFFFFF';

        // Top-Left corner
        ctx.beginPath();
        ctx.moveTo(x, y + bracketLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + bracketLen, y);
        ctx.stroke();

        // Top-Right corner
        ctx.beginPath();
        ctx.moveTo(x + width - bracketLen, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + bracketLen);
        ctx.stroke();

        // Bottom-Left corner
        ctx.beginPath();
        ctx.moveTo(x, y + height - bracketLen);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x + bracketLen, y + height);
        ctx.stroke();

        // Bottom-Right corner
        ctx.beginPath();
        ctx.moveTo(x + width - bracketLen, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width, y + height - bracketLen);
        ctx.stroke();

        // 4. Facial landmarks
        if (face.landmarks) {
          const points = [
            face.landmarks.left_eye,
            face.landmarks.right_eye,
            face.landmarks.nose_tip,
            face.landmarks.mouth_left || (face.landmarks as any).left_mouth_corner,
            face.landmarks.mouth_right || (face.landmarks as any).right_mouth_corner,
          ];

          const outerDotRadius = Math.max(3, Math.round(resScale * 4));
          const innerDotRadius = Math.max(1.5, Math.round(resScale * 1.8));

          points.forEach((pt) => {
            if (!pt) return;
            // Outer ring
            ctx.fillStyle = primaryColor;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, outerDotRadius, 0, 2 * Math.PI);
            ctx.fill();

            // Inner center dot
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, innerDotRadius, 0, 2 * Math.PI);
            ctx.fill();
          });
        }

        // 5. Clean, dark label pill
        const nameText = face.identity || face.matched_name || 'Unknown';
        const confPercent = ((face.confidence ?? 0) * 100).toFixed(1);
        const labelText = `${nameText} · ${confPercent}%`;

        const fontSize = Math.max(12, Math.round(resScale * 14));
        ctx.font = `600 ${fontSize}px Inter, sans-serif`;
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;

        const pillPaddingX = Math.round(resScale * 12);
        const pillHeight = Math.round(resScale * 28);
        const pillWidth = textWidth + pillPaddingX * 2 + Math.round(resScale * 12);
        const pillX = Math.max(0, x);
        const pillY = Math.max(0, y - pillHeight - Math.round(resScale * 6));

        // Draw pill background
        ctx.fillStyle = '#0F172A';
        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = Math.max(1.5, Math.round(resScale * 1.5));

        roundRect(ctx, pillX, pillY, pillWidth, pillHeight, Math.round(resScale * 5));
        ctx.fill();
        ctx.stroke();

        // Status circle dot
        const dotSize = Math.max(3, Math.round(resScale * 4));
        ctx.fillStyle = isKnown ? '#10B981' : '#F59E0B';
        ctx.beginPath();
        ctx.arc(pillX + pillPaddingX, pillY + pillHeight / 2, dotSize, 0, 2 * Math.PI);
        ctx.fill();

        // Label text
        ctx.fillStyle = '#F8FAFC';
        ctx.fillText(
          labelText,
          pillX + pillPaddingX + dotSize * 2,
          pillY + pillHeight / 2 + fontSize * 0.35
        );
      });
    };
  }, [imageSrc, faces, selectedFaceIndex]);

  // Click on canvas to select face
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSelectFace || faces.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Check which bounding box was clicked
    const clickedIdx = faces.findIndex((f) => {
      const fx = f.bbox.x1 ?? (f.bbox as any).x ?? 0;
      const fy = f.bbox.y1 ?? (f.bbox as any).y ?? 0;
      const fw = f.bbox.width ?? (f.bbox.x2 !== undefined ? f.bbox.x2 - fx : 100);
      const fh = f.bbox.height ?? (f.bbox.y2 !== undefined ? f.bbox.y2 - fy : 100);
      return clickX >= fx && clickX <= fx + fw && clickY >= fy && clickY <= fy + fh;
    });

    if (clickedIdx !== -1) {
      onSelectFace(clickedIdx);
    }
  };

  return (
    <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950 flex items-center justify-center">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="max-h-[640px] w-auto max-w-full object-contain cursor-crosshair"
      />
    </div>
  );
};

// Helper for drawing rounded rectangles
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

export default FaceCanvas;
