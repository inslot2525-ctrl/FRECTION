import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Shield, Users, AlertTriangle, Search, Loader2, Upload, FileText, ArrowLeft } from 'lucide-react'
import AnoAI from './AnoAI'
import FraudNetworkGraph from './FraudNetworkGraph'

const API = 'http://localhost:8000'

function fmt(n) {
  return typeof n === 'number' ? n.toLocaleString() : n
}

function MetricCard({ title, value, icon, color = 'text-white', loading }) {
  return (
    <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6 flex items-start justify-between">
      <div>
        <p className="text-gray-400 text-sm mb-1">{title}</p>
        {loading ? (
          <Loader2 className="animate-spin text-gray-500 mt-2" size={24} />
        ) : (
          <h3 className={`text-3xl font-light ${color}`}>{value ? fmt(value) : '—'}</h3>
        )}
      </div>
      <div className={`p-3 bg-white/5 rounded-lg ${color}`}>{icon}</div>
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [statusMessage, setStatusMessage] = useState('') // New state for live PyTorch logs
  const [stats, setStats] = useState(null)
  const [graphData, setGraphData] = useState(null)
  const [error, setError] = useState(null)
  const [graphKey, setGraphKey] = useState(0)

  // Handle file selection — immediately clear previous results so no grey overlay remains
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
      // Clear previous graph/metrics instantly — prevents grey overlapped view on replace
      setStats(null)
      setGraphData(null)
      setGraphKey((k) => k + 1)
    }
  }

  // Upload CSV and run fraud detection pipeline
  const handleUploadSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setUploading(true)
    setStatusMessage('Running GNN analysis...')
    setError(null)
    setStats(null)
    setGraphData(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API}/api/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Failed to process dataset.')
      }

      const data = await res.json()
      setStats(data.metrics)
      setGraphData(data.graph_data)
      setGraphKey((k) => k + 1)
      setStatusMessage('')
    } catch (err) {
      setError(err.message || 'Cannot reach the GNN backend server.')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="relative min-h-screen text-white font-sans bg-black">
      {/* Three.js Background Shader Layer */}
      <div className="absolute inset-0 z-0">
        <AnoAI />
      </div>

      <div className="relative z-10 p-8 min-h-screen bg-black/40 backdrop-blur-sm space-y-8">
        {/* Header */}
        <header className="flex justify-between items-center border-b border-white/10 pb-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/')}
              className="p-2 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white transition-all mr-1"
              title="Back to home"
            >
              <ArrowLeft size={18} />
            </button>
            <div
              onClick={() => navigate('/')}
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              title="Back to home"
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && navigate('/')}
            >
              <Shield className="text-cyan-400 w-8 h-8" />
              <h1 className="text-3xl font-light tracking-wider">
                FREC<span className="font-bold">TION</span> GNN ENGINE
              </h1>
            </div>
          </div>
          <span className="text-xs text-gray-400 border border-white/10 px-4 py-2 rounded-full bg-black/20">
            Pipeline: Upload &rarr; Detect &rarr; Visualize
          </span>
        </header>

        {/* Error Alert Banner */}
        {error && (
          <div className="p-4 rounded-xl border border-red-500/40 bg-red-500/10 text-red-300 text-sm">
            ⚠ {error}
          </div>
        )}

        {/* Upload Box Component */}
        <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
          <h2 className="text-xl font-medium mb-2">1. Upload Transaction Dataset</h2>
          <p className="text-gray-400 text-xs mb-4">
            Select your bank ledger CSV format containing sender/receiver mappings to run through the Graph Neural Network.
          </p>

          <form onSubmit={handleUploadSubmit} className="flex flex-col md:flex-row items-stretch md:items-center gap-4">
            <div className="flex-1 relative border border-dashed border-white/20 hover:border-cyan-400/50 rounded-xl p-4 flex items-center justify-center bg-black/30 transition-all cursor-pointer">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="flex items-center gap-3 text-sm text-gray-300">
                {file ? <FileText className="text-cyan-400 animate-pulse" size={20} /> : <Upload size={20} />}
                <span>{file ? file.name : 'Choose transaction_ledger.csv...'}</span>
              </div>
            </div>

            <button
              type="submit"
              disabled={!file || uploading}
              className="px-8 py-4 rounded-xl bg-cyan-500/20 border border-cyan-500/40 text-cyan-200 font-medium text-sm hover:bg-cyan-500/30 active:scale-[0.98] transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {uploading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Processing...
                </>
              ) : (
                'Execute Fraud Detection'
              )}
            </button>
          </form>
        </div>

        {/* Global Network Analytics Summary */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <MetricCard title="Total Accounts Scanned" value={stats?.total_nodes} icon={<Users size={20} />} loading={uploading} />
          <MetricCard title="Transactions Processed" value={stats?.total_edges} icon={<AlertTriangle size={20} />} loading={uploading} />
          <MetricCard title="Known Malicious Hubs" value={stats?.known_fraudsters} icon={<Shield size={20} />} loading={uploading} color="text-red-400" />
          <MetricCard title="Newly Identified Mules" value={stats?.suspected_mules} icon={<Search size={20} />} loading={uploading} color="text-cyan-400" />
        </div>

        {/* Main Graph Dynamic Container */}
        <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
          <div className="flex items-start justify-between gap-4 mb-1">
            <h2 className="text-xl font-medium">2. Network Topology & Risk Clusters</h2>
            {graphData && (
              <span className="shrink-0 inline-flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-cyan-300 border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 rounded-full">
                <Search size={12} className="text-cyan-400" /> Zoom in to see
              </span>
            )}
          </div>
          <p className="text-gray-400 text-xs mb-4">
            {graphData
              ? 'Interactive WebGL visualization generated from GNN node embeddings. • Scroll to zoom • Drag to pan'
              : 'Graph network structure will automatically generate here once a dataset has completed scanning.'}
          </p>

          <div className="rounded-xl overflow-hidden bg-black/40 border border-white/5 flex items-center justify-center min-h-[400px]">
            {graphData ? (
              <FraudNetworkGraph key={graphKey} graphData={graphData} />
            ) : (
              <div className="text-center p-8 space-y-2">
                {uploading ? (
                  <>
                    <Loader2 size={36} className="animate-spin mx-auto text-cyan-400 mb-2" />
                    <p className="text-sm font-mono text-cyan-300">{statusMessage || 'Constructing topological graph matrix...'}</p>
                  </>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-full border border-white/10 flex items-center justify-center mx-auto mb-2 text-gray-500">
                      <Search size={20} />
                    </div>
                    <p className="text-sm text-gray-400">Waiting for data payload upload...</p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}