import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  badge?: {
    text: string;
    variant?: 'brand' | 'success' | 'warning' | 'neutral';
  };
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
}) => {
  const getBadgeClass = (variant?: string) => {
    switch (variant) {
      case 'success':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'warning':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'brand':
        return 'bg-brand-500/10 text-brand-400 border-brand-500/20';
      default:
        return 'bg-slate-800 text-slate-400 border-slate-700';
    }
  };

  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 p-5 backdrop-blur-sm card-hover">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
          {title}
        </span>
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
          <Icon className="h-4 w-4 text-brand-500" />
        </div>
      </div>

      <div className="mt-3 flex items-baseline justify-between">
        <span className="text-2xl font-bold tracking-tight text-white">{value}</span>
        {badge && (
          <span
            className={`rounded-full px-2 py-0.5 text-[11px] font-semibold border ${getBadgeClass(
              badge.variant
            )}`}
          >
            {badge.text}
          </span>
        )}
      </div>

      {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
    </div>
  );
};

export default MetricCard;
