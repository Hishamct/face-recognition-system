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

      // Draw each face overlay
      faces.forEach((face, idx) => {
        const isSelected = idx === selectedFaceIndex;
        const { x, y, width, height } = face.bbox;

        // Colors (refined, non-neon)
        const isKnown =
          face.status === 'RECOGNIZED' ||
          face.is_known === true ||
          (Boolean(face.identity) && face.identity?.toUpperCase() !== 'UNKNOWN');
        const primaryColor = isKnown ? '#6366F1' : '#F59E0B'; // Indigo for known, Amber for unknown
        const fillColor = isKnown ? 'rgba(99, 102, 241, 0.08)' : 'rgba(245, 158, 11, 0.08)';

        // 1. Semi-transparent bounding box fill
        ctx.fillStyle = fillColor;
        ctx.fillRect(x, y, width, height);

        // 2. Bounding box border
        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(x, y, width, height);

        // 3. Technical corner brackets
        const bracketLen = Math.min(width, height) * 0.15;
        ctx.lineWidth = isSelected ? 4 : 3;
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

        // 4. Subtle 5-point facial landmarks
        if (face.landmarks) {
          const points = [
            face.landmarks.right_eye,
            face.landmarks.left_eye,
            face.landmarks.nose_tip,
            face.landmarks.right_mouth_corner,
            face.landmarks.left_mouth_corner,
          ];

          points.forEach((pt) => {
            if (!pt) return;
            // Outer point
            ctx.fillStyle = primaryColor;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 3.5, 0, 2 * Math.PI);
            ctx.fill();

            // Inner white dot
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 1.5, 0, 2 * Math.PI);
            ctx.fill();
          });
        }

        // 5. Clean, dark label pill
        const nameText = face.identity || face.matched_name || 'Unknown';
        const confPercent = ((face.confidence ?? 0) * 100).toFixed(1);
        const labelText = `${nameText} · ${confPercent}%`;

        ctx.font = '500 13px Inter, sans-serif';
        const textMetrics = ctx.measureText(labelText);
        const textWidth = textMetrics.width;
        const pillHeight = 26;
        const pillWidth = textWidth + 20;
        const pillX = Math.max(0, x);
        const pillY = Math.max(0, y - pillHeight - 6);

        // Draw pill background
        ctx.fillStyle = '#0F172A';
        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = 1;

        // Rounded rect for label
        roundRect(ctx, pillX, pillY, pillWidth, pillHeight, 4);
        ctx.fill();
        ctx.stroke();

        // Status circle dot
        ctx.fillStyle = isKnown ? '#10B981' : '#F59E0B';
        ctx.beginPath();
        ctx.arc(pillX + 10, pillY + pillHeight / 2, 4, 0, 2 * Math.PI);
        ctx.fill();

        // Text
        ctx.fillStyle = '#F8FAFC';
        ctx.fillText(labelText, pillX + 18, pillY + 17);
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
    const clickedIdx = faces.findIndex(
      (f) =>
        clickX >= f.bbox.x &&
        clickX <= f.bbox.x + f.bbox.width &&
        clickY >= f.bbox.y &&
        clickY <= f.bbox.y + f.bbox.height
    );

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
