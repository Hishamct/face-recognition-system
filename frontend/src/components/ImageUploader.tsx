import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  onClear?: () => void;
  selectedFile?: File | null;
  loading?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelected,
  onClear,
  selectedFile,
  loading = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageSelected(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      onImageSelected(file);
    }
  };

  return (
    <div className="w-full">
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
            isDragging
              ? 'border-brand-500 bg-brand-500/5'
              : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/60'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="p-3 mb-3 rounded-full bg-slate-800/80 border border-slate-700/60 text-slate-300">
            <UploadCloud className="h-6 w-6 text-brand-500" />
          </div>
          <p className="text-sm font-medium text-slate-200">
            Click to upload or drag & drop an image
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Supports JPEG, PNG, or WebP (max 15 MB)
          </p>
        </div>
      ) : (
        <div className="flex items-center justify-between p-3.5 rounded-lg bg-slate-900/80 border border-slate-800">
          <div className="flex items-center space-x-3 overflow-hidden">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-slate-800 border border-slate-700 text-brand-500">
              <ImageIcon className="h-5 w-5" />
            </div>
            <div className="truncate">
              <p className="text-sm font-medium text-slate-200 truncate">{selectedFile.name}</p>
              <p className="text-xs text-slate-500">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB · {selectedFile.type}
              </p>
            </div>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (onClear) onClear();
              if (inputRef.current) inputRef.current.value = '';
            }}
            disabled={loading}
            className="ml-3 p-1.5 rounded-md text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors"
            title="Remove image"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
