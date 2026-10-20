import React, { createContext, useContext, useState, ReactNode } from 'react';
import type { RecognitionResponse } from '../types/api.types';

interface AnalysisContextType {
  selectedFile: File | null;
  previewUrl: string | null;
  result: RecognitionResponse | null;
  selectedFaceIdx: number;
  similarityThreshold: number;
  loading: boolean;
  error: string | null;
  setSelectedFile: (file: File | null) => void;
  setPreviewUrl: (url: string | null) => void;
  setResult: (result: RecognitionResponse | null) => void;
  setSelectedFaceIdx: (idx: number) => void;
  setSimilarityThreshold: (threshold: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined);

export const AnalysisProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [selectedFaceIdx, setSelectedFaceIdx] = useState<number>(0);
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(0.57);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const clearAnalysis = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setResult(null);
    setSelectedFaceIdx(0);
    setError(null);
    setLoading(false);
  };

  return (
    <AnalysisContext.Provider
      value={{
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
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = (): AnalysisContextType => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};

export default AnalysisContext;
