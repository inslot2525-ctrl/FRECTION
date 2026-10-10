import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Shield, Users, AlertTriangle, Search, Loader2, Upload, FileText, ArrowLeft,
  X, Sparkles, CheckCircle2, Clock, Columns3, Info,
} from 'lucide-react'
import AnoAI from './AnoAI'
import FraudNetworkGraph from './FraudNetworkGraph'
import AccountInvestigator from './AccountInvestigator'

// Same-origin by default: Vite proxies /api to the FastAPI server on :8000.
// Set VITE_API_URL to point at a backend elsewhere.
const API = import.meta.env.VITE_API_URL ?? ''

const OFFLINE_MSG =
  'Cannot reach the analysis server. Start the backend (.\\start.ps1, or: uvicorn api.main:app --port 8000) and try again.'

// Add an account's neighbourhood to the drawn graph (skipping what's already there)
function mergeGraph(prev, extra) {
  if (!prev) return prev
  const have = new Set(prev.nodes.map((n) => n.id))
  const linkKey = (l) => `${l.source}→${l.target}`
  const haveLinks = new Set(prev.links.map((l) => linkKey({
    source: typeof l.source === 'object' ? l.source.id : l.source,
    target: typeof l.target === 'object' ? l.target.id : l.target,
  })))
  const nodes = extra.nodes.filter((n) => !have.has(n.id))
  const links = extra.links.filter((l) => !haveLinks.has(linkKey(l)))
  if (!nodes.length && !links.length) return prev
  return { nodes: [...prev.nodes, ...nodes], links: [...prev.links, ...links] }
}

function fmt(n) {
  return typeof n === 'number' ? n.toLocaleString() : n
}

function fmtBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 ** 2).toFixed(1)} MB`
}

function MetricCard({ title, value, icon, color = 'text-white', loading }) {
  return (
    <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6 flex items-start justify-between">
      <div>
        <p className="text-gray-400 text-sm mb-1">{title}</p>
        {loading ? (
          <Loader2 className="animate-spin text-gray-500 mt-2" size={24} />
        ) : (
          <h3 className={`text-3xl font-light ${color}`}>{value != null ? fmt(value) : '—'}</h3>
        )}
      </div>
      <div className={`p-3 bg-white/5 rounded-lg ${color}`}>{icon}</div>
    </div>
  )
}

function ServerStatus({ status }) {
  const styles = {
    ready:       { dot: 'bg-emerald-400', text: 'Engine ready',            title: 'Graph rules and GNN risk model loaded' },
    unavailable: { dot: 'bg-amber-400',   text: 'Rules only',              title: 'GNN risk model not installed — graph rules, records mode and explanations all work' },
    loading:     { dot: 'bg-cyan-400 animate-pulse', text: 'Warming up model…', title: 'The GNN model is loading in the background' },
    offline:     { dot: 'bg-red-500',     text: 'Backend offline',         title: OFFLINE_MSG },
    checking:    { dot: 'bg-gray-500 animate-pulse', text: 'Connecting…', title: '' },
  }[status]
  return (
    <span title={styles.title} className="inline-flex items-center gap-2 text-xs text-gray-300 border border-white/10 px-3 py-1.5 rounded-full bg-black/30">
      <span className={`w-2 h-2 rounded-full ${styles.dot}`} />
      {styles.text}
    </span>
  )
}

const NONE = '__none__'

const ENGINE_TEXT = {
  'structural': 'graph rules',
  'structural+gnn-score': 'graph rules + GNN risk score',
  'gnn+structural': 'graph rules + GNN lookup',
}

const MODE_TEXT = {
  transactions: {
    chip: 'Transaction network',
    cards: ['Total Accounts Scanned', 'Transactions Processed', 'Known Malicious Hubs', 'Newly Identified Mules'],
    graph: 'accounts and transfers, prioritising flagged activity',
    legend: { fraud: 'Fraud actor', mule: 'Mule / offshore', normal: 'Normal' },
  },
  entities: {
    chip: 'Customer records',
    cards: ['Customers Analysed', 'Records Processed', 'Fraud (labelled / most anomalous)', 'Newly Suspected Look-alikes'],
    graph: 'customers linked to their most similar peers — tight clusters are potential rings',
    legend: { fraud: 'Fraud / highly anomalous', mule: 'Suspected look-alike', normal: 'Normal' },
  },
}

// Form fields the backend accepts to override auto-detected columns
function overridesFromDraft(d) {
  if (!d) return {}
  const base = { mode: d.mode, label_col: d.label || NONE }
  return d.mode === 'transactions'
    ? { ...base, sender_col: d.sender, receiver_col: d.receiver, amount_col: d.amount || NONE }
    : { ...base, id_col: d.id || NONE }
}

function ColumnSelect({ label, value, onChange, columns, optional = true }) {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-gray-400 min-w-[150px] flex-1">
      {label}
      <select
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        className="bg-black/60 border border-white/15 rounded-lg px-2 py-2 text-xs text-gray-200 focus:border-cyan-400/60 outline-none"
      >
        {optional ? <option value="">— none —</option> : <option value="" disabled>Choose a column…</option>}
        {columns.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
    </label>
  )
}

// POST the file with upload progress (fetch can't report upload progress)
function uploadCsv(file, overrides, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API}/api/analyze`)
    xhr.responseType = 'json'
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.upload.onload = () => onProgress(1)
    xhr.onload = () => {
      const body = xhr.response
      if (xhr.status >= 200 && xhr.status < 300 && body) return resolve(body)
      if (xhr.status >= 500 && !body) return reject(new Error(OFFLINE_MSG))
      reject(new Error(body?.detail || `Failed to process dataset (HTTP ${xhr.status}).`))
    }
    xhr.onerror = () => reject(new Error(OFFLINE_MSG))
    const formData = new FormData()
    formData.append('file', file)
    Object.entries(overrides).forEach(([k, v]) => v && formData.append(k, v))
    xhr.send(formData)
  })
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [stage, setStage] = useState('')          // live progress text
  const [progress, setProgress] = useState(0)     // upload progress 0..1
  const [stats, setStats] = useState(null)
  const [graphData, setGraphData] = useState(null)
  const [result, setResult] = useState(null)      // column_mapping + meta
  const [error, setError] = useState(null)
  const [graphKey, setGraphKey] = useState(0)
  const [dragging, setDragging] = useState(false)
  const [server, setServer] = useState('checking')
  const [columns, setColumns] = useState([])      // every column in the uploaded file
  const [draft, setDraft] = useState(null)        // editable column mapping
  const [showEditor, setShowEditor] = useState(false)
  const [analysisId, setAnalysisId] = useState(null)
  const [aiEnabled, setAiEnabled] = useState(false)
  const [selectedAccount, setSelectedAccount] = useState(null)
  const inputRef = useRef(null)

  // Poll backend health until the model is ready (or known to be unavailable)
  useEffect(() => {
    let cancelled = false
    let timer
    const check = async () => {
      let next = 'offline'
      try {
        const res = await fetch(`${API}/api/health`)
        if (res.ok) next = (await res.json()).model ?? 'ready'
      } catch { /* offline */ }
      if (cancelled) return
      setServer(next)
      if (next === 'loading' || next === 'offline') timer = setTimeout(check, next === 'loading' ? 1500 : 4000)
    }
    check()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [])

  const resetResults = () => {
    setError(null)
    setStats(null)
    setGraphData(null)
    setResult(null)
    setAnalysisId(null)
    setSelectedAccount(null)
    setGraphKey((k) => k + 1)
  }

  const forgetColumns = () => {
    setColumns([])
    setDraft(null)
    setShowEditor(false)
  }

  const selectFile = (f) => {
    if (!f) return
    resetResults()
    forgetColumns()
    if (!f.name.toLowerCase().endsWith('.csv')) {
      setFile(null)
      setError(`"${f.name}" is not a CSV file. Please choose a .csv transaction ledger.`)
      return
    }
    setFile(f)
  }

  const runAnalysis = useCallback(async (f, overrides = {}) => {
    if (!f || uploading) return
    setUploading(true)
    resetResults()
    setProgress(0)
    setStage('Uploading dataset…')

    try {
      const data = await uploadCsv(f, overrides, (p) => {
        setProgress(p)
        if (p >= 1) setStage('Understanding columns and running analysis…')
      })
      setStage('Rendering network…')
      const m = data.column_mapping
      setStats(data.metrics)
      setGraphData(data.graph_data)
      setResult({ columns: m, meta: data.meta, mode: data.mode })
      setColumns(data.columns ?? [])
      setAnalysisId(data.analysis_id ?? null)
      setAiEnabled(Boolean(data.ai_explanations))
      setDraft({ mode: data.mode, sender: m.sender, receiver: m.receiver, amount: m.amount, label: m.fraud, id: m.id })
      setGraphKey((k) => k + 1)
    } catch (err) {
      setError(err.message || OFFLINE_MSG)
    } finally {
      setUploading(false)
      setStage('')
    }
  }, [uploading])

  const handleUploadSubmit = (e) => {
    e.preventDefault()
    runAnalysis(file, draft && draftValid ? overridesFromDraft(draft) : {})
  }

  const trySample = async () => {
    try {
      const res = await fetch('/sample_ledger.csv')
      if (!res.ok) throw new Error()
      const sample = new File([await res.blob()], 'sample_ledger.csv', { type: 'text/csv' })
      setFile(sample)
      runAnalysis(sample)
    } catch {
      setError('Could not load the sample dataset.')
    }
  }

  const clearFile = (e) => {
    e.stopPropagation()
    setFile(null)
    resetResults()
    forgetColumns()
    if (inputRef.current) inputRef.current.value = ''
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    if (uploading) return
    selectFile(e.dataTransfer.files?.[0])
  }

  // Select an account; if it isn't drawn yet, pull in its direct connections first
  const selectAccount = useCallback(async (id) => {
    setSelectedAccount(id)
    if (!id || !analysisId || graphData?.nodes.some((n) => n.id === id)) return
    try {
      const res = await fetch(`${API}/api/analysis/${analysisId}/neighbourhood?${new URLSearchParams({ account: id })}`)
      if (res.ok) {
        const extra = await res.json()
        setGraphData((prev) => mergeGraph(prev, extra))
      }
    } catch { /* the investigator panel still shows the evidence */ }
  }, [analysisId, graphData])

  // Demo link: /detect?demo=1 loads the sample; &account=<id> also opens that account
  const demoStarted = useRef(false)
  const demoAccountOpened = useRef(false)
  useEffect(() => {
    if (demoStarted.current || !new URLSearchParams(window.location.search).has('demo')) return
    demoStarted.current = true
    trySample()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  useEffect(() => {
    const account = new URLSearchParams(window.location.search).get('account')
    if (!account || !analysisId || !graphData || demoAccountOpened.current) return
    demoAccountOpened.current = true
    selectAccount(account)
  }, [analysisId, graphData, selectAccount])

  const mapping = result?.columns
  const meta = result?.meta
  const text = MODE_TEXT[result?.mode] ?? MODE_TEXT.transactions
  const setDraftField = (field) => (value) => setDraft((d) => ({ ...d, [field]: value }))
  const draftValid = draft && (draft.mode !== 'transactions' || (draft.sender && draft.receiver && draft.sender !== draft.receiver))

  return (
    <div className="relative min-h-screen text-white font-sans bg-black">
      {/* Three.js Background Shader Layer */}
      <div className="fixed inset-0 z-0">
        <AnoAI />
      </div>

      <div className="relative z-10 p-4 sm:p-8 min-h-screen bg-black/40 space-y-8">
        {/* Header */}
        <header className="flex flex-wrap gap-3 justify-between items-center border-b border-white/10 pb-4">
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
              <h1 className="text-2xl sm:text-3xl font-light tracking-wider">
                FREC<span className="font-bold">TION</span> GNN ENGINE
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <ServerStatus status={server} />
            <span className="hidden md:inline text-xs text-gray-400 border border-white/10 px-4 py-2 rounded-full bg-black/20">
              Pipeline: Upload &rarr; Detect &rarr; Visualize
            </span>
          </div>
        </header>

        {/* Error Alert Banner */}
        {error && (
          <div role="alert" className="flex items-start justify-between gap-4 p-4 rounded-xl border border-red-500/40 bg-red-500/10 text-red-300 text-sm">
            <span>⚠ {error}</span>
            <button onClick={() => setError(null)} className="text-red-300/70 hover:text-red-200" title="Dismiss">
              <X size={16} />
            </button>
          </div>
        )}

        {/* Upload Box Component */}
        <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
            <div>
              <h2 className="text-xl font-medium mb-2">1. Upload Transaction Dataset</h2>
              <p className="text-gray-400 text-xs">
                Drop any CSV — a transaction ledger (sender → receiver) or a customer table (one row per customer). Columns, delimiters and formats are detected automatically.
              </p>
            </div>
            <button
              type="button"
              onClick={trySample}
              disabled={uploading}
              className="inline-flex items-center gap-2 text-xs text-cyan-300 border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 rounded-lg hover:bg-cyan-500/20 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Sparkles size={14} /> No CSV handy? Try sample data
            </button>
          </div>

          <form onSubmit={handleUploadSubmit} className="flex flex-col md:flex-row items-stretch md:items-center gap-4">
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => !uploading && inputRef.current?.click()}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
              role="button"
              tabIndex={0}
              className={`flex-1 relative border border-dashed rounded-xl p-4 flex items-center justify-center transition-all cursor-pointer ${
                dragging ? 'border-cyan-400 bg-cyan-500/10' : 'border-white/20 hover:border-cyan-400/50 bg-black/30'
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => selectFile(e.target.files?.[0])}
                className="hidden"
              />
              <div className="flex items-center gap-3 text-sm text-gray-300 min-w-0">
                {file ? <FileText className="text-cyan-400 shrink-0" size={20} /> : <Upload size={20} className="shrink-0" />}
                {file ? (
                  <>
                    <span className="truncate">{file.name}</span>
                    <span className="text-gray-500 text-xs shrink-0">{fmtBytes(file.size)}</span>
                    {!uploading && (
                      <button type="button" onClick={clearFile} className="p-1 rounded hover:bg-white/10 text-gray-500 hover:text-white shrink-0" title="Remove file">
                        <X size={14} />
                      </button>
                    )}
                  </>
                ) : (
                  <span>{dragging ? 'Drop it here' : 'Drag & drop a CSV here, or click to browse'}</span>
                )}
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
              ) : stats ? (
                'Re-run Detection'
              ) : (
                'Execute Fraud Detection'
              )}
            </button>
          </form>

          {/* Result summary: what was detected and how fast */}
          {meta && mapping && (
            <div className="mt-4 flex flex-wrap items-center gap-2 text-[11px] text-gray-300">
              <span className="inline-flex items-center gap-1.5 text-emerald-300 border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 rounded-full">
                <CheckCircle2 size={12} /> Analysis complete
              </span>
              <span className="inline-flex items-center gap-1.5 border border-white/10 bg-black/30 px-2.5 py-1 rounded-full">
                <Clock size={12} /> {meta.elapsed_ms < 1000 ? `${meta.elapsed_ms} ms` : `${(meta.elapsed_ms / 1000).toFixed(1)} s`}
                {' · '}{ENGINE_TEXT[meta.engine] ?? 'anomaly detection + similarity graph'}
              </span>
              <span className="inline-flex items-center gap-1.5 text-cyan-300 border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-1 rounded-full">
                {text.chip}
              </span>
              <span className="inline-flex flex-wrap items-center gap-1.5 border border-white/10 bg-black/30 px-2.5 py-1 rounded-full">
                <Columns3 size={12} /> Columns:
                {result.mode === 'transactions' ? (
                  <>
                    <span className="text-gray-400">sender</span> {mapping.sender}
                    <span className="text-gray-400">· receiver</span> {mapping.receiver}
                    {mapping.amount && <><span className="text-gray-400">· amount</span> {mapping.amount}</>}
                  </>
                ) : (
                  <>
                    <span className="text-gray-400">customer ID</span> {mapping.id ?? 'row number'}
                    <span className="text-gray-400">· features</span> {meta.features_count}
                  </>
                )}
                <span className="text-gray-400">· label</span> {mapping.fraud ?? 'none (unsupervised)'}
              </span>
              <button
                type="button"
                onClick={() => setShowEditor((v) => !v)}
                className="inline-flex items-center gap-1.5 text-cyan-300 hover:text-cyan-200 underline underline-offset-2"
              >
                {showEditor ? 'Hide column settings' : 'Wrong columns? Adjust'}
              </button>
              {meta.rows_truncated && (
                <span className="inline-flex items-center gap-1.5 text-amber-300 border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 rounded-full">
                  <Info size={12} /> Large file — first {fmt(meta.max_rows)} rows analysed
                </span>
              )}
            </div>
          )}

          {/* Column editor: lets users correct the auto-detection and re-run */}
          {showEditor && draft && columns.length > 0 && (
            <div className="mt-4 p-4 rounded-xl border border-white/10 bg-black/40 space-y-4">
              <div className="flex flex-wrap gap-2 text-xs">
                {[
                  ['transactions', 'Transactions', 'each row is a transfer from one account to another'],
                  ['entities', 'Customer records', 'each row describes one customer / account'],
                ].map(([value, title, desc]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setDraftField('mode')(value)}
                    className={`text-left px-3 py-2 rounded-lg border transition-all ${
                      draft.mode === value ? 'border-cyan-400/60 bg-cyan-500/15 text-cyan-100' : 'border-white/10 text-gray-400 hover:border-white/25'
                    }`}
                  >
                    <span className="block font-medium">{title}</span>
                    <span className="block text-[11px] opacity-70">{desc}</span>
                  </button>
                ))}
              </div>

              <div className="flex flex-wrap gap-3">
                {draft.mode === 'transactions' ? (
                  <>
                    <ColumnSelect label="Sender account" value={draft.sender} onChange={setDraftField('sender')} columns={columns} optional={false} />
                    <ColumnSelect label="Receiver account" value={draft.receiver} onChange={setDraftField('receiver')} columns={columns} optional={false} />
                    <ColumnSelect label="Amount (optional)" value={draft.amount} onChange={setDraftField('amount')} columns={columns} />
                  </>
                ) : (
                  <ColumnSelect label="Customer ID (optional — row numbers otherwise)" value={draft.id} onChange={setDraftField('id')} columns={columns} />
                )}
                <ColumnSelect label="Fraud / class label (optional)" value={draft.label} onChange={setDraftField('label')} columns={columns} />
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={!draftValid || uploading}
                  onClick={() => runAnalysis(file, overridesFromDraft(draft))}
                  className="px-4 py-2 rounded-lg bg-cyan-500/20 border border-cyan-500/40 text-cyan-200 text-xs font-medium hover:bg-cyan-500/30 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Re-run with these columns
                </button>
                <span className="text-[11px] text-gray-500">
                  {draft.mode === 'transactions'
                    ? draftValid ? 'Accounts become nodes, transfers become edges.' : 'Pick two different columns for sender and receiver.'
                    : 'All other numeric / categorical columns are normalised and used as features.'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Global Network Analytics Summary */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard title={text.cards[0]} value={stats?.total_nodes} icon={<Users size={20} />} loading={uploading} />
          <MetricCard title={text.cards[1]} value={stats?.total_edges} icon={<AlertTriangle size={20} />} loading={uploading} />
          <MetricCard title={text.cards[2]} value={stats?.known_fraudsters} icon={<Shield size={20} />} loading={uploading} color="text-red-400" />
          <MetricCard title={text.cards[3]} value={stats?.suspected_mules} icon={<Search size={20} />} loading={uploading} color="text-cyan-400" />
        </div>

        {/* Graph + account investigator */}
        <div className={`grid grid-cols-1 gap-6 ${graphData && analysisId ? 'xl:grid-cols-[minmax(0,1fr)_400px]' : ''}`}>
        <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6 min-w-0">
          <div className="flex flex-wrap items-start justify-between gap-4 mb-1">
            <h2 className="text-xl font-medium">2. Network Topology & Risk Clusters</h2>
            {graphData && (
              <span className="shrink-0 inline-flex items-center gap-1.5 text-[11px] font-medium tracking-wide text-cyan-300 border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5 rounded-full max-w-[320px] text-center leading-tight">
                <Search size={12} className="text-cyan-400 shrink-0" /> Zoom in and hover to see account names — click a node to see why it was flagged
              </span>
            )}
          </div>
          <p className="text-gray-400 text-xs mb-4">
            {graphData
              ? `Showing ${fmt(graphData.nodes.length)} nodes and ${fmt(graphData.links.length)} links: ${text.graph}. • Scroll to zoom • Drag to pan • Hover over nodes to see names`
              : 'Graph network structure will automatically generate here once a dataset has completed scanning.'}
          </p>

          <div className="rounded-xl overflow-hidden bg-black/40 border border-white/5 flex items-center justify-center min-h-[480px]">
            {graphData ? (
              <FraudNetworkGraph
                key={graphKey}
                graphData={graphData}
                labels={text.legend}
                selectedId={selectedAccount}
                onNodeClick={selectAccount}
              />
            ) : (
              <div className="text-center p-8 space-y-2">
                {uploading ? (
                  <>
                    <Loader2 size={36} className="animate-spin mx-auto text-cyan-400 mb-2" />
                    <p className="text-sm font-mono text-cyan-300">{stage || 'Constructing topological graph matrix...'}</p>
                    {progress < 1 && file && file.size > 512 * 1024 && (
                      <div className="w-56 h-1.5 mx-auto rounded-full bg-white/10 overflow-hidden">
                        <div className="h-full bg-cyan-400 transition-all" style={{ width: `${Math.round(progress * 100)}%` }} />
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-full border border-white/10 flex items-center justify-center mx-auto mb-2 text-gray-500">
                      <Search size={20} />
                    </div>
                    <p className="text-sm text-gray-400">Waiting for data payload upload...</p>
                    <p className="text-xs text-gray-500">Drop a CSV above, or try the sample dataset to see it in action.</p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {graphData && analysisId && (
          <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6 flex flex-col xl:h-[640px] max-h-[80vh] xl:max-h-none">
            <AccountInvestigator
              analysisId={analysisId}
              aiEnabled={aiEnabled}
              labels={text.legend}
              selected={selectedAccount}
              onSelect={selectAccount}
            />
          </div>
        )}
        </div>
      </div>
    </div>
  )
}
