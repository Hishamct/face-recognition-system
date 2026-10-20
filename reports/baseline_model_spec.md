# Baseline Recognition Model & Evaluation Specification
======================================================

## 1. Architectural Strategy: Pretrained Baseline First

Rather than training a deep convolutional or transformer backbone from scratch on 4,079 images, our engineering roadmap establishes a **pretrained baseline first**:

```
                       ┌──────────────────────────────────────────────┐
                       │          Phase 3: Pretrained Baseline        │
                       │  (InsightFace ArcFace / OpenCV SFace ONNX)   │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Zero-Shot Evaluation on Validation Set     │
                       │   (Calibrate Similarity Threshold tau on val)│
                       └──────────────────────┬───────────────────────┘
                                              │
                       ┌──────────────────────┴───────────────────────┐
                       ▼                                              ▼
       ┌───────────────────────────────┐              ┌───────────────────────────────┐
       │   Branch A: Satisfies Goals   │              │   Branch B: Needs Improvement │
       │   Deploy directly into vector │              │   Fine-tune head / ArcFace    │
       │   search database (pgvector)  │              │   margin loss in Phase 4      │
       └───────────────────────────────┘              └───────────────────────────────┘
```

### Pretrained Backbone Selection:
1. **Primary Model**: **ArcFace ResNet-50 / MobileFaceNet** (512-dimensional $\ell_2$-normalized embedding vector).
2. **Alternative / Lightweight Fallback**: **OpenCV SFace** (128-dimensional embedding vector, built into OpenCV DNN).
3. **Inputs**: $112 \times 112 \times 3$ aligned RGB face crop (generated in Phase 2).

---

## 2. Evaluation Metrics Protocol

The system evaluates three distinct real-world tasks across the partitions:

### Task 1: Closed-Set Identification (1:N Gallery Search)
- **Goal**: When a test probe photo of known identity $k$ is queried against the enrolled gallery (`train/`), is identity $k$ returned at rank 1?
- **Dataset Partition**: Gallery built from `data/processed/train/`; probes drawn from `data/processed/test/`.
- **Metrics**:
  - **Rank-1 Accuracy**: Percentage of probe queries where the top nearest neighbor is the true identity.
  - **Top-5 Accuracy**: Percentage of probe queries where the true identity is within the top 5 nearest neighbors.
  - **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the true identity across all probe queries ($\frac{1}{Q} \sum_{i=1}^Q \frac{1}{\text{rank}_i}$).

### Task 2: Open-Set Identification & Unknown Rejection
- **Goal**: When an unenrolled intruder or unknown person is probed against the gallery, does the system reject them instead of assigning a false identity?
- **Threshold Calibration**: The rejection threshold $\tau$ is calibrated **strictly on `data/processed/val/`** to target a specific operating point (e.g., $\text{FAR} \le 1.0\%$).
- **Metrics**:
  - **False Acceptance Rate (FAR)**: $\frac{\text{False Matches}}{\text{Total Impostor Queries}}$ (security metric).
  - **False Rejection Rate (FRR)**: $\frac{\text{False Non-Matches}}{\text{Total Genuine Queries}}$ (usability metric).
  - **ROC-AUC**: Area under the Receiver Operating Characteristic curve across varying thresholds $\tau$.

### Task 3: Pairwise Verification (1:1 Comparison)
- **Goal**: Determine whether two arbitrary face crops represent the same person: $\text{sim}(e_1, e_2) \ge \tau$.
- **Metrics**:
  - **Cosine Similarity Verification Accuracy**: Accuracy at the optimal validation-calibrated threshold.
  - **F1-Score**: Harmonic mean of Precision and Recall on balanced genuine/impostor pairs.
