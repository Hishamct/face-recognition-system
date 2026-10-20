import axios from 'axios';
import type {
  DetectionResponse,
  RecognitionResponse,
  RecognitionAuditItem,
  IdentityListResponse,
  IdentityResponse,
  IdentityEnrollResponse,
  IdentityDeleteResponse,
  EvaluationMetricsResponse,
  TrainingRunsResponse,
  TrainingHistoryResponse,
  HealthResponse,
  VideoSummaryResponse,
} from '../types/api.types';

// Axios instance configured to use Vite proxy or localhost directly
const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
});

export const api = {
  // Health
  checkHealth: async (): Promise<HealthResponse> => {
    const res = await apiClient.get<HealthResponse>('/health');
    return res.data;
  },

  // Face Detection
  detectFaces: async (
    file: File,
    scoreThreshold = 0.8,
    nmsThreshold = 0.3
  ): Promise<DetectionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post<DetectionResponse>('/detection/image', formData, {
      params: { score_threshold: scoreThreshold, nms_threshold: nmsThreshold },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Face Recognition
  recognizeFaces: async (
    file: File,
    similarityThreshold?: number,
    topK = 5
  ): Promise<RecognitionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const params: Record<string, any> = { top_k: topK };
    if (similarityThreshold !== undefined) {
      params.similarity_threshold = similarityThreshold;
    }
    const res = await apiClient.post<RecognitionResponse>('/recognition/image', formData, {
      params,
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Recognition Audit Logs
  getAuditLogs: async (limit = 50): Promise<RecognitionAuditItem[]> => {
    const res = await apiClient.get<RecognitionAuditItem[]>('/recognition/audit-logs', {
      params: { limit },
    });
    return res.data;
  },

  // Identity Management
  listIdentities: async (skip = 0, limit = 50): Promise<IdentityListResponse> => {
    const res = await apiClient.get<IdentityListResponse>('/identities', {
      params: { skip, limit },
    });
    return res.data;
  },

  getIdentity: async (id: number): Promise<IdentityResponse> => {
    const res = await apiClient.get<IdentityResponse>(`/identities/${id}`);
    return res.data;
  },

  enrollIdentity: async (
    name: string,
    department?: string,
    notes?: string,
    file?: File
  ): Promise<IdentityEnrollResponse> => {
    const formData = new FormData();
    formData.append('name', name);
    if (department) formData.append('department', department);
    if (notes) formData.append('notes', notes);
    if (file) formData.append('file', file);

    const res = await apiClient.post<IdentityEnrollResponse>('/identities', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  deleteIdentity: async (id: number): Promise<IdentityDeleteResponse> => {
    const res = await apiClient.delete<IdentityDeleteResponse>(`/identities/${id}`);
    return res.data;
  },

  // Evaluation Benchmarks
  getEvaluationMetrics: async (): Promise<EvaluationMetricsResponse> => {
    const res = await apiClient.get<EvaluationMetricsResponse>('/evaluation/metrics');
    return res.data;
  },

  getEvaluationHistory: async (): Promise<EvaluationMetricsResponse[]> => {
    const res = await apiClient.get<EvaluationMetricsResponse[]>('/evaluation/history');
    return res.data;
  },

  // Training Runs & History
  getTrainingRuns: async (): Promise<TrainingRunsResponse> => {
    const res = await apiClient.get<TrainingRunsResponse>('/training/runs');
    return res.data;
  },

  getTrainingHistory: async (runId?: number): Promise<TrainingHistoryResponse> => {
    const res = await apiClient.get<TrainingHistoryResponse>('/training/history', {
      params: runId !== undefined ? { run_id: runId } : {},
    });
    return res.data;
  },

  // Video Stream & Batch Video Processing (Phase 7)
  uploadVideo: async (
    file: File,
    stride = 5,
    confidenceThreshold = 0.5697,
    maxFrames = 300
  ): Promise<VideoSummaryResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post<VideoSummaryResponse>('/video/upload', formData, {
      params: {
        stride,
        confidence_threshold: confidenceThreshold,
        max_frames: maxFrames,
      },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  getVideoWebSocketUrl: (): string => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // When running under Vite proxy or localhost, route to /api/v1/video/stream
    return `${protocol}//${window.location.host}/api/v1/video/stream`;
  },
};

export default api;
