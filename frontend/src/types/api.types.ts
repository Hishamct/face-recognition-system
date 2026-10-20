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
  face_id: number;
  bbox: BoundingBox;
  confidence: number;
  landmarks: Landmarks5;
}

export interface DetectionResponse {
  total_faces: number;
  image_width: number;
  image_height: number;
  inference_time_ms: number;
  faces: FaceDetectionItem[];
}

export interface CandidateItem {
  identity_name: string;
  similarity: number;
  probability?: number | null;
}

export interface FaceRecognitionItem {
  face_id: number;
  bbox: BoundingBox;
  landmarks: Landmarks5;
  matched_name: string;
  confidence: number;
  is_known: boolean;
  threshold_used: number;
  candidates: CandidateItem[];
}

export interface RecognitionResponse {
  total_faces: number;
  image_width: number;
  image_height: number;
  inference_time_ms: number;
  faces: FaceRecognitionItem[];
}

export interface RecognitionAuditItem {
  id: number;
  timestamp: string;
  face_count: number;
  matched_identity?: string | null;
  top_confidence?: number | null;
  is_known?: boolean | null;
  inference_time_ms?: number | null;
}

export interface IdentityResponse {
  id: number;
  name: string;
  department?: string | null;
  notes?: string | null;
  exemplar_count: number;
  has_prototype: boolean;
  created_at: string;
  updated_at: string;
}

export interface IdentityListResponse {
  total: number;
  skip: number;
  limit: number;
  identities: IdentityResponse[];
}

export interface IdentityEnrollResponse {
  message: string;
  identity: IdentityResponse;
}

export interface IdentityDeleteResponse {
  message: string;
  identity_id: number;
  deleted_embeddings_count: number;
}

export interface EvaluationMetricsResponse {
  benchmark_name: string;
  dataset_partition: string;
  total_pairs?: number | null;
  genuine_pairs?: number | null;
  impostor_pairs?: number | null;
  calibrated_threshold: number;
  fpir: number;
  fnir: number;
  tar_at_far_01?: number | null;
  tar_at_far_001?: number | null;
  eer?: number | null;
  auc?: number | null;
  timestamp: string;
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
  environment: string;
  version: string;
  database?: string;
}
