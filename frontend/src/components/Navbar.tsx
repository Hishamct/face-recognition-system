import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { ScanFace, Users, LineChart, Video, ShieldCheck, AlertCircle } from 'lucide-react';
import api from '../services/api';

export const Navbar: React.FC = () => {
  const [dbStatus, setDbStatus] = useState<'connected' | 'error' | 'checking'>('checking');

  useEffect(() => {
    let isMounted = true;
    const checkConnection = async () => {
      try {
        const res = await api.checkHealth();
        if (isMounted) {
          if (
            res.status === 'healthy' ||
            res.status === 'ok' ||
            res.database === 'connected' ||
            Boolean((res as any).database_connected)
          ) {
            setDbStatus('connected');
          } else {
            setDbStatus('error');
          }
        }
      } catch (err) {
        if (isMounted) setDbStatus('error');
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { to: '/', label: 'Image Analysis', icon: ScanFace },
    { to: '/identities', label: 'Identities', icon: Users },
    { to: '/analytics', label: 'Analytics & Training', icon: LineChart },
    { to: '/live', label: 'Live Stream', icon: Video, badge: 'Phase 7' },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-dark-950/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Brand Logo */}
        <div className="flex items-center space-x-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-900 border border-slate-800 text-brand-500">
            <ScanFace className="h-6 w-6 text-brand-500" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-base font-semibold tracking-tight text-white">FaceIntel</span>
              <span className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] font-medium text-slate-400 border border-slate-700/50">
                PROD
              </span>
            </div>
            <p className="text-xs text-slate-400">Biometric Recognition Platform</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center space-x-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center space-x-2 rounded-md px-3.5 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-slate-800/90 text-white border border-slate-700/60 shadow-sm'
                      : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                  }`
                }
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
                {item.badge && (
                  <span className="ml-1 rounded bg-slate-800 px-1.5 py-0.2 text-[9px] font-semibold text-slate-400 border border-slate-700">
                    {item.badge}
                  </span>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* System Health Status Indicator */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 rounded-full bg-slate-900/90 px-3 py-1 text-xs border border-slate-800">
            {dbStatus === 'connected' ? (
              <>
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-medium text-slate-300">Backend Live</span>
              </>
            ) : dbStatus === 'checking' ? (
              <>
                <span className="h-2 w-2 rounded-full bg-amber-500" />
                <span className="text-slate-400">Connecting...</span>
              </>
            ) : (
              <>
                <span className="h-2 w-2 rounded-full bg-rose-500" />
                <span className="text-rose-400">Backend Offline</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
