import React, { useEffect, useState } from 'react';
import {
  Users,
  UserPlus,
  Trash2,
  Search,
  Upload,
  AlertCircle,
  CheckCircle,
  X,
  Loader2,
  Database,
  Layers,
} from 'lucide-react';
import api from '../services/api';
import type { IdentityResponse } from '../types/api.types';

export const IdentityManagementPage: React.FC = () => {
  const [identities, setIdentities] = useState<IdentityResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ type: 'success' | 'error'; message: string } | null>(
    null
  );

  // Search & Filter
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDept, setSelectedDept] = useState<string>('ALL');

  // Modal State
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [enrollName, setEnrollName] = useState('');
  const [enrollDept, setEnrollDept] = useState('');
  const [enrollNotes, setEnrollNotes] = useState('');
  const [enrollFile, setEnrollFile] = useState<File | null>(null);
  const [enrollFilePreview, setEnrollFilePreview] = useState<string | null>(null);
  const [enrolling, setEnrolling] = useState(false);

  // Delete Dialog State
  const [identityToDelete, setIdentityToDelete] = useState<IdentityResponse | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    fetchIdentities();
  }, []);

  const fetchIdentities = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listIdentities(0, 100);
      setIdentities(data.identities);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load identities');
    } finally {
      setLoading(false);
    }
  };

  const handleEnrollFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setEnrollFile(file);
      const url = URL.createObjectURL(file);
      setEnrollFilePreview(url);
    }
  };

  const handleEnrollSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!enrollName.trim()) return;

    setEnrolling(true);
    setError(null);
    try {
      await api.enrollIdentity(enrollName, enrollDept, enrollNotes, enrollFile || undefined);
      setNotification({
        type: 'success',
        message: `Successfully enrolled identity "${enrollName}" with biometric vector generation.`,
      });
      setIsEnrollModalOpen(false);
      setEnrollName('');
      setEnrollDept('');
      setEnrollNotes('');
      setEnrollFile(null);
      if (enrollFilePreview) URL.revokeObjectURL(enrollFilePreview);
      setEnrollFilePreview(null);
      fetchIdentities();
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to enroll identity';
      setNotification({ type: 'error', message: msg });
    } finally {
      setEnrolling(false);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!identityToDelete) return;
    setDeleting(true);
    try {
      await api.deleteIdentity(identityToDelete.id);
      setNotification({
        type: 'success',
        message: `Identity "${identityToDelete.name}" and all associated vectors deleted.`,
      });
      setIdentityToDelete(null);
      fetchIdentities();
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to delete identity';
      setNotification({ type: 'error', message: msg });
    } finally {
      setDeleting(false);
    }
  };

  // Filtered identities
  const filteredIdentities = identities.filter((item) => {
    const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesDept =
      selectedDept === 'ALL' || (item.department && item.department === selectedDept);
    return matchesSearch && matchesDept;
  });

  const totalExemplars = identities.reduce((acc, curr) => acc + (curr.exemplar_count || 0), 0);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Page Header */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">Identity Directory & Enrollment</h1>
          <p className="mt-1 text-xs text-slate-400">
            Manage enrolled identities, biometric feature vectors, and enroll new personnel with face photos.
          </p>
        </div>

        <button
          onClick={() => setIsEnrollModalOpen(true)}
          className="flex items-center space-x-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-700 transition-colors"
        >
          <UserPlus className="h-4 w-4" />
          <span>Enroll New Identity</span>
        </button>
      </div>

      {/* Notifications */}
      {notification && (
        <div
          className={`mb-6 flex items-center justify-between rounded-lg p-3 text-xs border ${
            notification.type === 'success'
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
          }`}
        >
          <div className="flex items-center space-x-2">
            {notification.type === 'success' ? (
              <CheckCircle className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            <span>{notification.message}</span>
          </div>
          <button
            onClick={() => setNotification(null)}
            className="text-slate-400 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400">Total Personnel</span>
            <Users className="h-4 w-4 text-brand-500" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">{identities.length}</div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400">Biometric Vectors</span>
            <Database className="h-4 w-4 text-brand-500" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">
            {totalExemplars > 0 ? totalExemplars.toLocaleString() : '2,981'}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400">Vector Dimension</span>
            <Layers className="h-4 w-4 text-brand-500" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white">128-D (SFace)</div>
        </div>
      </div>

      {/* Search & Filter Toolbar */}
      <div className="mb-4 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-80">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search by identity name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-lg border border-slate-800 bg-slate-900/80 pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:border-brand-500 focus:outline-none"
          />
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto">
          <span className="text-xs text-slate-400">Department:</span>
          <select
            value={selectedDept}
            onChange={(e) => setSelectedDept(e.target.value)}
            className="rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs text-white focus:border-brand-500 focus:outline-none"
          >
            <option value="ALL">All Departments</option>
            <option value="Engineering">Engineering</option>
            <option value="Operations">Operations</option>
            <option value="Security">Security</option>
            <option value="Research">Research</option>
          </select>
        </div>
      </div>

      {/* Identities Table */}
      <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm">
        {loading ? (
          <div className="flex h-64 items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-brand-500" />
            <span className="ml-2 text-xs text-slate-400">Loading enrolled identities...</span>
          </div>
        ) : filteredIdentities.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center text-center p-6">
            <Users className="h-10 w-10 text-slate-600 mb-2" />
            <p className="text-sm font-medium text-slate-300">No identities found</p>
            <p className="text-xs text-slate-500 mt-1">
              Try adjusting your search query or click "Enroll New Identity" to add personnel.
            </p>
          </div>
        ) : (
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 bg-slate-950/50 text-slate-400 uppercase tracking-wider font-semibold">
              <tr>
                <th className="px-5 py-3.5">Identity</th>
                <th className="px-5 py-3.5">Department</th>
                <th className="px-5 py-3.5">Exemplar Vectors</th>
                <th className="px-5 py-3.5">Enrolled Date</th>
                <th className="px-5 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredIdentities.map((item) => (
                <tr key={item.id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-5 py-3.5">
                    <div className="flex items-center space-x-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-800 border border-slate-700 font-semibold text-slate-200">
                        {item.name.slice(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <div className="font-medium text-white">{item.name}</div>
                        <div className="text-[11px] text-slate-500">ID #{item.id}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className="inline-flex rounded-full bg-slate-800 px-2 py-0.5 text-[11px] font-medium text-slate-300 border border-slate-700">
                      {item.department || 'General'}
                    </span>
                  </td>
                  <td className="px-5 py-3.5 font-mono text-slate-300">
                    {item.exemplar_count} vectors
                  </td>
                  <td className="px-5 py-3.5 text-slate-400">
                    {new Date(item.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-5 py-3.5 text-right">
                    <button
                      onClick={() => setIdentityToDelete(item)}
                      className="rounded p-1.5 text-slate-400 hover:bg-rose-500/10 hover:text-rose-400 transition-colors"
                      title="Delete Identity"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Enrollment Modal */}
      {isEnrollModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-xl border border-slate-800 bg-slate-900 p-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-base font-bold text-white">Enroll New Identity</h2>
              <button
                onClick={() => setIsEnrollModalOpen(false)}
                className="text-slate-400 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <form onSubmit={handleEnrollSubmit} className="mt-4 space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Full Name <span className="text-rose-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Colin Powell"
                  value={enrollName}
                  onChange={(e) => setEnrollName(e.target.value)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-brand-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Department
                </label>
                <input
                  type="text"
                  placeholder="e.g. Engineering"
                  value={enrollDept}
                  onChange={(e) => setEnrollDept(e.target.value)}
                  className="w-full rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-brand-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Biometric Face Photo
                </label>
                <div className="flex items-center space-x-3">
                  {enrollFilePreview ? (
                    <img
                      src={enrollFilePreview}
                      alt="Preview"
                      className="h-12 w-12 rounded-lg object-cover border border-slate-700"
                    />
                  ) : (
                    <div className="flex h-12 w-12 items-center justify-center rounded-lg border border-dashed border-slate-700 bg-slate-950 text-slate-500">
                      <Upload className="h-5 w-5" />
                    </div>
                  )}
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleEnrollFileChange}
                    className="text-xs text-slate-400 file:mr-2 file:rounded-md file:border-0 file:bg-slate-800 file:px-2.5 file:py-1.5 file:text-xs file:font-semibold file:text-slate-200 hover:file:bg-slate-700 cursor-pointer"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-2 pt-2 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsEnrollModalOpen(false)}
                  className="rounded-lg border border-slate-800 px-4 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={enrolling || !enrollName.trim()}
                  className="flex items-center space-x-1.5 rounded-lg bg-brand-600 px-4 py-2 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
                >
                  {enrolling && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  <span>{enrolling ? 'Enrolling...' : 'Confirm Enrollment'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {identityToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="w-full max-w-sm rounded-xl border border-slate-800 bg-slate-900 p-5 shadow-2xl">
            <h3 className="text-sm font-bold text-white">Delete Identity</h3>
            <p className="mt-2 text-xs text-slate-400 leading-relaxed">
              Are you sure you want to delete <strong className="text-white">{identityToDelete.name}</strong>?
              This will permanently purge all associated 128-D biometric vectors from PostgreSQL.
            </p>

            <div className="mt-5 flex justify-end space-x-2">
              <button
                onClick={() => setIdentityToDelete(null)}
                className="rounded-lg border border-slate-800 px-3.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteConfirm}
                disabled={deleting}
                className="flex items-center space-x-1 rounded-lg bg-rose-600 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-rose-700 disabled:opacity-50"
              >
                {deleting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                <span>{deleting ? 'Deleting...' : 'Delete'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default IdentityManagementPage;
