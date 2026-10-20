import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import type { EpochMetricItem } from '../types/api.types';

interface LossAccuracyChartProps {
  epochs: EpochMetricItem[];
}

export const LossAccuracyChart: React.FC<LossAccuracyChartProps> = ({ epochs }) => {
  const [metricTab, setMetricTab] = useState<'loss' | 'accuracy'>('loss');

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 gap-2">
        <div>
          <h3 className="text-sm font-semibold text-white">Epoch Training Progression</h3>
          <p className="text-xs text-slate-500">Supervised IdentityClassifier head metrics</p>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center space-x-1 rounded-lg bg-slate-800/80 p-0.5 border border-slate-700/60">
          <button
            onClick={() => setMetricTab('loss')}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              metricTab === 'loss'
                ? 'bg-slate-700 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Loss Curves
          </button>
          <button
            onClick={() => setMetricTab('accuracy')}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              metricTab === 'accuracy'
                ? 'bg-slate-700 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Accuracy Curves
          </button>
        </div>
      </div>

      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={epochs} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
            <XAxis
              dataKey="epoch"
              stroke="#64748B"
              fontSize={12}
              tickLine={false}
              tickFormatter={(v) => `Ep ${v}`}
            />
            <YAxis
              stroke="#64748B"
              fontSize={12}
              tickLine={false}
              domain={metricTab === 'loss' ? ['auto', 'auto'] : [0.7, 1.0]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0F172A',
                borderColor: '#334155',
                borderRadius: '0.5rem',
                fontSize: '12px',
                color: '#F8FAFC',
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }}
              iconType="circle"
            />
            {metricTab === 'loss' ? (
              <>
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  name="Train Loss"
                  stroke="#38BDF8"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#38BDF8' }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  name="Validation Loss"
                  stroke="#818CF8"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#818CF8' }}
                  activeDot={{ r: 5 }}
                />
              </>
            ) : (
              <>
                <Line
                  type="monotone"
                  dataKey="train_acc"
                  name="Train Accuracy"
                  stroke="#34D399"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#34D399' }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  type="monotone"
                  dataKey="val_acc"
                  name="Validation Accuracy"
                  stroke="#6366F1"
                  strokeWidth={2}
                  dot={{ r: 3, fill: '#6366F1' }}
                  activeDot={{ r: 5 }}
                />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default LossAccuracyChart;
