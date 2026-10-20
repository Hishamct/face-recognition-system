export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
}

export interface Landmarks5 {
  right_eye: LandmarkPoint;
  left_eye: LandmarkPoint;
  nose_tip: LandmarkPoint;
  right_mouth_corner: LandmarkPoint;
  left_mouth_corner: LandmarkPoint;
}

export interface FaceDetectionItem {
  face_id?: number;
  bbox: BoundingBox;
  confidence: number;
  landmarks: Landmarks5;
}

export interface DetectionResponse {
  success?: boolean;
  face_count?: number;
  total_faces?: number;
  image_width: number;
  image_height: number;
  latency_ms?: number;
  inference_time_ms?: number;
  faces: FaceDetectionItem[];
}

export interface CandidateItem {
  identity?: string;
  identity_name?: string;
  confidence?: number;
  similarity?: number;
  probability?: number | null;
  rank?: number;
}

export interface FaceRecognitionItem {
  face_id?: number;
  bbox: BoundingBox;
  landmarks: Landmarks5;
  identity?: string;
  matched_name?: string;
  confidence: number;
  status?: string; // 'RECOGNIZED' | 'UNKNOWN'
  is_known?: boolean;
  threshold_used: number;
  top_candidates?: CandidateItem[];
  candidates?: CandidateItem[];
}

export interface RecognitionResponse {
  success?: boolean;
  face_count?: number;
  total_faces?: number;
  recognized_count?: number;
  unknown_count?: number;
  image_width: number;
  image_height: number;
  latency_ms?: number;
  inference_time_ms?: number;
  faces: FaceRecognitionItem[];
}

export interface RecognitionAuditItem {
  id: number;
  timestamp?: string;
  created_at?: string;
  face_count?: number;
  matched_identity?: string | null;
  similarity_score?: number | null;
  top_confidence?: number | null;
  is_known?: boolean | null;
  decision?: string | null;
  query_latency_ms?: number | null;
  inference_time_ms?: number | null;
}

export interface IdentityResponse {
  id: number;
  name: string;
  department?: string | null;
  notes?: string | null;
  is_active?: boolean;
  embedding_count?: number;
  exemplar_count?: number;
  has_prototype?: boolean;
  created_at: string;
  updated_at?: string;
}

export interface IdentityListResponse {
  total: number;
  page?: number;
  page_size?: number;
  skip?: number;
  limit?: number;
  items?: IdentityResponse[];
  identities?: IdentityResponse[];
}

export interface IdentityEnrollResponse {
  success?: boolean;
  message: string;
  identity_id?: number;
  name?: string;
  identity?: IdentityResponse;
  vector_id?: number;
  embedding_count?: number;
}

export interface IdentityDeleteResponse {
  success?: boolean;
  message: string;
  identity_id: number;
  deleted_embeddings_count?: number;
}

export interface EvaluationMetricsResponse {
  id?: number | null;
  evaluation_name?: string;
  benchmark_name?: string;
  dataset_partition?: string;
  model_version?: string;
  gallery_version?: string;
  total_pairs?: number | null;
  genuine_pairs?: number | null;
  impostor_pairs?: number | null;
  threshold_used?: number;
  calibrated_threshold?: number;
  rank1_acc?: number;
  top5_acc?: number;
  top10_acc?: number | null;
  mrr?: number;
  tpir?: number;
  fpir: number;
  fnir?: number;
  tar_at_far_01?: number | null;
  tar_at_far_001?: number | null;
  eer?: number | null;
  auc?: number | null;
  macro_auc?: number | null;
  micro_auc?: number | null;
  tested_at?: string | null;
  timestamp?: string;
}

export interface TrainingRunItem {
  id: number;
  run_name: string;
  backbone: string;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  dropout?: number | null;
  weight_decay?: number | null;
  early_stopped_epoch?: number | null;
  best_val_loss?: number | null;
  best_val_acc?: number | null;
  status: string;
  created_at: string;
}

export interface TrainingRunsResponse {
  total: number;
  runs: TrainingRunItem[];
}

export interface EpochMetricItem {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  learning_rate?: number | null;
}

export interface TrainingHistoryResponse {
  run: TrainingRunItem;
  epochs: EpochMetricItem[];
}

export interface HealthResponse {
  status: string;
  environment?: string;
  version?: string;
  database?: string;
}
