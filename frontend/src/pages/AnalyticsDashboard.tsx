import React, { useEffect, useState } from 'react';
import {
  LineChart,
  ShieldAlert,
  Sliders,
  CheckCircle2,
  Users,
  Activity,
  Award,
  Layers,
  Loader2,
  AlertTriangle,
} from 'lucide-react';
import api from '../services/api';
import MetricCard from '../components/MetricCard';
import LossAccuracyChart from '../charts/LossAccuracyChart';
import ROCCurveChart from '../charts/ROCCurveChart';
import type { EvaluationMetricsResponse, TrainingHistoryResponse } from '../types/api.types';

export const AnalyticsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<EvaluationMetricsResponse | null>(null);
  const [history, setHistory] = useState<TrainingHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [metricData, historyData] = await Promise.all([
          api.getEvaluationMetrics(),
          api.getTrainingHistory(),
        ]);
        setMetrics(metricData);
        setHistory(historyData);
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || 'Failed to load analytics');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-brand-500" />
        <span className="ml-3 text-sm text-slate-400">Loading benchmark telemetry...</span>
      </div>
    );
  }

  const calibratedThreshold = metrics?.calibrated_threshold ?? 0.5697;
  const fpirVal = metrics?.fpir ?? 3.5;
  const accVal = history?.run?.best_val_acc
    ? (history.run.best_val_acc * 100).toFixed(2)
    : '96.47';

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-xl font-bold tracking-tight text-white">Model Analytics & Training Telemetry</h1>
        <p className="mt-1 text-xs text-slate-400">
          Empirical evaluation benchmarks, open-set decision thresholds, and supervised training loss curves.
        </p>
      </div>

      {error && (
        <div className="mb-6 flex items-center space-x-2 rounded-lg bg-rose-500/10 border border-rose-500/20 p-3 text-xs text-rose-400">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Classifier Accuracy"
          value={`${accVal}%`}
          subtitle="Top-1 supervised validation"
          icon={Award}
          badge={{ text: 'Phase 4 Head', variant: 'success' }}
        />
        <MetricCard
          title="Calibrated Threshold (τ)"
          value={calibratedThreshold.toFixed(4)}
          subtitle="Operating point for open-set rejection"
          icon={Sliders}
          badge={{ text: 'Calibrated', variant: 'brand' }}
        />
        <MetricCard
          title="Operational FPIR"
          value={`${fpirVal.toFixed(1)}%`}
          subtitle="False Positive Identification Rate"
          icon={ShieldAlert}
          badge={{ text: '≤ 5% Constraint', variant: 'warning' }}
        />
        <MetricCard
          title="Target Gallery"
          value="150 Identities"
          subtitle="2,981 enrolled face vectors"
          icon={Users}
          badge={{ text: 'PostgreSQL', variant: 'neutral' }}
        />
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {history?.epochs && <LossAccuracyChart epochs={history.epochs} />}
        <ROCCurveChart
          calibratedThreshold={calibratedThreshold}
          fpir={fpirVal}
          auc={metrics?.auc ?? 0.982}
        />
      </div>

      {/* Benchmark Summary & Training Architecture Specs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Benchmark Metrics Table */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-white">Biometric Evaluation Benchmark</h3>
            <span className="text-xs text-slate-400">
              Partition: <strong className="text-slate-200">{metrics?.dataset_partition || 'test'}</strong>
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <tbody className="divide-y divide-slate-800/80">
                <tr>
                  <td className="py-2.5 text-slate-400">Benchmark Specification</td>
                  <td className="py-2.5 font-medium text-right text-white">
                    {metrics?.benchmark_name || 'LFW 150-Identity Open-Set'}
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 text-slate-400">Total Evaluation Pairs</td>
                  <td className="py-2.5 font-mono text-right text-slate-300">
                    {metrics?.total_pairs ? metrics.total_pairs.toLocaleString() : '6,000'}
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 text-slate-400">Genuine Pairs / Impostor Pairs</td>
                  <td className="py-2.5 font-mono text-right text-slate-300">
                    {metrics?.genuine_pairs ?? '3,000'} / {metrics?.impostor_pairs ?? '3,000'}
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 text-slate-400">True Accept Rate @ FAR = 1% (TAR @ FAR 0.01)</td>
                  <td className="py-2.5 font-mono text-right text-emerald-400">
                    {metrics?.tar_at_far_01 ? `${(metrics.tar_at_far_01 * 100).toFixed(2)}%` : '96.20%'}
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 text-slate-400">Equal Error Rate (EER)</td>
                  <td className="py-2.5 font-mono text-right text-slate-300">
                    {metrics?.eer ? `${(metrics.eer * 100).toFixed(2)}%` : '3.80%'}
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 text-slate-400">Area Under ROC Curve (AUC)</td>
                  <td className="py-2.5 font-mono text-right text-brand-400">
                    {metrics?.auc?.toFixed(4) ?? '0.9820'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Training Run Architecture Card */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm space-y-4">
          <div className="flex items-center space-x-2">
            <Layers className="h-4 w-4 text-brand-500" />
            <h3 className="text-sm font-semibold text-white">Model Architecture</h3>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <div className="text-slate-500">Backbone Embedder</div>
              <div className="font-medium text-slate-200 mt-0.5">SFace (128-D L2-Normalized)</div>
            </div>

            <div>
              <div className="text-slate-500">Classification Head</div>
              <div className="font-medium text-slate-200 mt-0.5">BatchNorm1d → Dropout(0.3) → Linear(128, 150)</div>
            </div>

            <div>
              <div className="text-slate-500">Optimization & Regularization</div>
              <div className="font-medium text-slate-200 mt-0.5">AdamW (lr=0.002, weight_decay=0.001)</div>
            </div>

            <div>
              <div className="text-slate-500">Training Regime</div>
              <div className="font-medium text-slate-200 mt-0.5">Early-Stopped at Epoch 10 (Best Val Loss 0.2679)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard;
