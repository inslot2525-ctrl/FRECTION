import { useNavigate } from 'react-router-dom'
import { Shield, Network, Search, ArrowRight, GitBranch, Zap, Eye } from 'lucide-react'
import AnoAI from './AnoAI'

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-6 flex flex-col gap-3 hover:bg-white/8 hover:border-white/20 transition-all duration-300">
      <div className="p-3 bg-white/5 rounded-xl w-fit text-cyan-400">
        {icon}
      </div>
      <h3 className="text-white font-medium text-base">{title}</h3>
      <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
    </div>
  )
}

function Step({ number, title, desc }) {
  return (
    <div className="flex gap-4 items-start">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 text-sm font-medium">
        {number}
      </div>
      <div>
        <p className="text-white text-sm font-medium mb-1">{title}</p>
        <p className="text-gray-400 text-xs leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

export default function LandingPage() {
  const navigate = useNavigate()

  return (
    <div className="relative min-h-screen text-white font-sans bg-black">
      {/* Animated background */}
      <div className="fixed inset-0 z-0">
        <AnoAI />
      </div>

      <div className="relative z-10 min-h-screen bg-black/50">

        {/* ── HEADER ── */}
        <header className="flex justify-between items-center px-8 py-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <Shield className="text-cyan-400 w-7 h-7" />
            <span className="text-2xl font-light tracking-wider">
              FREC<span className="font-bold">TION</span>
              <span className="text-gray-400 font-light text-base ml-2">GNN ENGINE</span>
            </span>
          </div>
          <button
            onClick={() => navigate('/detect')}
            className="text-sm text-cyan-400 border border-cyan-500/40 px-4 py-2 rounded-full hover:bg-cyan-500/10 transition-all"
          >
            Launch App →
          </button>
        </header>

        {/* ── HERO ── */}
        <section className="px-8 pt-20 pb-16 max-w-4xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 text-xs text-cyan-400 border border-cyan-500/30 bg-cyan-500/10 px-4 py-1.5 rounded-full mb-2">
            <Zap size={12} />
            Powered by Graph Neural Networks
          </div>

          <h1 className="text-5xl md:text-6xl font-light leading-tight tracking-tight">
            Find fraud rings<br />
            <span className="text-cyan-400 font-semibold">hiding in plain sight</span>
          </h1>

          <p className="text-gray-300 text-lg max-w-2xl mx-auto leading-relaxed">
            Upload any bank transaction CSV. FRECTION maps every account as a node,
            every transfer as an edge, and surfaces coordinated fraud rings your
            rule-based systems miss entirely.
          </p>

          <button
            onClick={() => navigate('/detect')}
            className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-cyan-500/20 border border-cyan-500/50 text-cyan-200 font-medium text-base hover:bg-cyan-500/30 active:scale-[0.98] transition-all duration-200 mt-4"
          >
            Try it out
            <ArrowRight size={18} />
          </button>

          <p className="text-gray-500 text-xs pt-2">
            No account needed · Works with transaction ledgers or customer tables · Results in seconds
          </p>
        </section>

        {/* ── HOW IT WORKS ── */}
        <section className="px-8 py-16 max-w-5xl mx-auto">
          <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-8">
            <h2 className="text-xl font-medium mb-8 text-center">How it works</h2>

            {/* Mini transaction flow diagram */}
            <div className="flex flex-wrap items-center justify-center gap-2 text-xs mb-10 font-mono">
              {[
                { label: 'V2001', color: 'text-gray-400 border-white/20', dot: 'bg-gray-500' },
                { label: '→ $500', color: 'text-gray-500', dot: null },
                { label: 'F3001', color: 'text-red-400 border-red-500/40', dot: 'bg-red-500' },
                { label: '→ $900', color: 'text-gray-500', dot: null },
                { label: 'C439737079', color: 'text-cyan-400 border-cyan-500/40', dot: 'bg-cyan-400' },
                { label: '→ $3800', color: 'text-gray-500', dot: null },
                { label: 'OFFSHORE_999', color: 'text-cyan-400 border-cyan-500/40', dot: 'bg-cyan-400' },
              ].map((item, i) =>
                item.dot ? (
                  <span key={i} className={`flex items-center gap-1.5 border px-3 py-1 rounded-full ${item.color}`}>
                    <span className={`w-2 h-2 rounded-full ${item.dot}`} />
                    {item.label}
                  </span>
                ) : (
                  <span key={i} className={item.color}>{item.label}</span>
                )
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Step
                number="1"
                title="Upload your CSV"
                desc="Any transaction file with sender and receiver columns. The engine auto-detects column names — no formatting required."
              />
              <Step
                number="2"
                title="Graph analysis runs"
                desc="Every account becomes a node. Every transfer becomes an edge. The GNN scans for fan-in patterns, layering, and cash-out signatures."
              />
              <Step
                number="3"
                title="Fraud ring visualised"
                desc="Red nodes are fraud actors. Cyan nodes are mule hubs and offshore accounts. Grey nodes are clean. The full network is interactive."
              />
            </div>
          </div>
        </section>

        {/* ── FEATURES ── */}
        <section className="px-8 py-4 pb-16 max-w-5xl mx-auto">
          <h2 className="text-xl font-medium mb-8 text-center">What gets detected</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <FeatureCard
              icon={<Network size={20} />}
              title="Mule hub detection"
              desc="Accounts that receive from many senders and funnel to one destination — the consolidation layer of any fraud ring."
            />
            <FeatureCard
              icon={<Search size={20} />}
              title="Fraud actor tracing"
              desc="Accounts that harvest from victims and route funds upstream. Detected structurally, even without labelled training data."
            />
            <FeatureCard
              icon={<GitBranch size={20} />}
              title="Layering & cash-out"
              desc="Shell companies and offshore accounts at the end of the chain are found by following the money downstream from a mule hub."
            />
            <FeatureCard
              icon={<Eye size={20} />}
              title="No labels needed"
              desc="Works on raw unlabelled CSVs. The graph structure itself is the signal — no isFraud column required."
            />
            <FeatureCard
              icon={<Zap size={20} />}
              title="GNN embeddings"
              desc="A pre-trained Graph Neural Network adds deep pattern matching on top of structural heuristics for higher accuracy."
            />
            <FeatureCard
              icon={<Shield size={20} />}
              title="Any CSV format"
              desc="Transaction ledgers or one-row-per-customer tables. Columns, delimiters, encodings, currencies and Yes/No labels are understood automatically — and you can adjust them."
            />
          </div>
        </section>

        {/* ── CTA ── */}
        <section className="px-8 py-16 text-center">
          <div className="bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl p-12 max-w-2xl mx-auto space-y-5">
            <Shield className="text-cyan-400 mx-auto" size={36} />
            <h2 className="text-2xl font-light">Ready to scan your data?</h2>
            <p className="text-gray-400 text-sm">
              Upload a CSV and get a full fraud ring map in seconds.
            </p>
            <button
              onClick={() => navigate('/detect')}
              className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-cyan-500/20 border border-cyan-500/50 text-cyan-200 font-medium hover:bg-cyan-500/30 active:scale-[0.98] transition-all duration-200"
            >
              Try it out
              <ArrowRight size={18} />
            </button>
          </div>
        </section>

        {/* ── FOOTER ── */}
        <footer className="border-t border-white/10 px-8 py-6 text-center text-xs text-gray-600">
          FRECTION · Graph Neural Network fraud detection
        </footer>
      </div>
    </div>
  )
}
