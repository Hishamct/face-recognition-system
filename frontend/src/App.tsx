import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import ImageAnalysisPage from './pages/ImageAnalysisPage';
import IdentityManagementPage from './pages/IdentityManagementPage';
import AnalyticsDashboard from './pages/AnalyticsDashboard';
import LiveVideoPage from './pages/LiveVideoPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-dark-950 text-slate-100 font-sans selection:bg-brand-600 selection:text-white">
        <Navbar />

        <main className="flex-1 pb-16">
          <Routes>
            <Route path="/" element={<ImageAnalysisPage />} />
            <Route path="/identities" element={<IdentityManagementPage />} />
            <Route path="/analytics" element={<AnalyticsDashboard />} />
            <Route path="/live" element={<LiveVideoPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <footer className="border-t border-slate-800/80 bg-dark-950/80 py-4 text-center text-xs text-slate-500">
          <div className="mx-auto max-w-7xl px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <div>Face Recognition System · Production Biometric Pipeline</div>
            <div className="flex items-center space-x-4 text-slate-400">
              <span>YuNet Detector</span>
              <span>•</span>
              <span>SFace 128-D</span>
              <span>•</span>
              <span>PostgreSQL & pgvector</span>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
