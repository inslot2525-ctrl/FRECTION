import { useEffect, useRef, useState } from 'react'
import {
  Search, ChevronDown, Loader2, ShieldAlert, Users, Info, Sparkles, Send, X, ArrowRight, KeyRound,
} from 'lucide-react'

/**
 * AccountInvestigator.jsx
 * "Why is this account flagged?" — pick an account (click a node, search, or
 * choose from the flagged list) and see the exact evidence behind its verdict.
 * If the backend has an Anthropic API key, an AI write-up + follow-up Q&A is offered.
 */

const API = import.meta.env.VITE_API_URL ?? ''

const GROUP_STYLE = {
  fraud:  { badge: 'text-red-300 border-red-500/40 bg-red-500/10',   dot: 'bg-red-500' },
  mule:   { badge: 'text-cyan-300 border-cyan-500/40 bg-cyan-500/10', dot: 'bg-cyan-400' },
  normal: { badge: 'text-gray-300 border-white/15 bg-white/5',        dot: 'bg-gray-500' },
}

const KIND_ICON = {
  fraud: <ShieldAlert size={14} className="text-red-400 shrink-0 mt-0.5" />,
  mule:  <Users size={14} className="text-cyan-400 shrink-0 mt-0.5" />,
  info:  <Info size={14} className="text-gray-400 shrink-0 mt-0.5" />,
}

async function getJson(url, options) {
  const res = await fetch(url, options)
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new Error(body?.detail || `Request failed (HTTP ${res.status}).`)
  return body
}

// Minimal, safe renderer for the AI's markdown-ish text: paragraphs, bullets, **bold**
function RichText({ text }) {
  const bold = (line) => line.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? <strong key={i} className="text-white">{part.slice(2, -2)}</strong> : part)
  const blocks = []
  let bullets = []
  const flush = () => {
    if (bullets.length) blocks.push(<ul key={blocks.length} className="list-disc pl-5 space-y-1">{bullets}</ul>)
    bullets = []
  }
  text.split('\n').forEach((raw) => {
    const line = raw.trim()
    if (/^[-*•]\s+/.test(line)) bullets.push(<li key={bullets.length}>{bold(line.replace(/^[-*•]\s+/, ''))}</li>)
    else {
      flush()
      if (line) blocks.push(<p key={blocks.length}>{bold(line.replace(/^#+\s*/, ''))}</p>)
    }
  })
  flush()
  return <div className="space-y-2">{blocks}</div>
}

function AccountPicker({ analysisId, labels, onSelect }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState('fraud')
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const boxRef = useRef(null)

  // Search (debounced) or list flagged accounts of the selected tab
  useEffect(() => {
    if (!open || !analysisId) return
    let cancelled = false
    const t = setTimeout(async () => {
      setLoading(true)
      try {
        const params = new URLSearchParams(query ? { q: query, limit: 40 } : { group: tab, limit: 200 })
        const data = await getJson(`${API}/api/analysis/${analysisId}/accounts?${params}`)
        if (!cancelled) { setItems(data.accounts); setTotal(data.total) }
      } catch {
        if (!cancelled) { setItems([]); setTotal(0) }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }, query ? 180 : 0)
    return () => { cancelled = true; clearTimeout(t) }
  }, [query, tab, open, analysisId])

  // Close when clicking outside
  useEffect(() => {
    const onDown = (e) => boxRef.current && !boxRef.current.contains(e.target) && setOpen(false)
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [])

  const choose = (id) => {
    onSelect(id)
    setOpen(false)
    setQuery('')
  }

  // Enter: look the typed text up directly, so it works even before the debounced list loads
  const chooseFirstMatch = async () => {
    if (!query.trim()) return items[0] && choose(items[0].id)
    try {
      const params = new URLSearchParams({ q: query.trim(), limit: 1 })
      const data = await getJson(`${API}/api/analysis/${analysisId}/accounts?${params}`)
      if (data.accounts[0]) choose(data.accounts[0].id)
    } catch { /* keep the dropdown open */ }
  }

  return (
    <div ref={boxRef} className="relative">
      <div className="flex items-center gap-2 bg-black/50 border border-white/15 rounded-lg px-3 py-2 focus-within:border-cyan-400/60">
        <Search size={14} className="text-gray-500 shrink-0" />
        <input
          value={query}
          onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') chooseFirstMatch()
            if (e.key === 'Escape') setOpen(false)
          }}
          placeholder="Search any account, or pick a flagged one…"
          className="flex-1 min-w-0 bg-transparent text-xs text-gray-200 placeholder:text-gray-500 outline-none"
        />
        <button type="button" onClick={() => setOpen((o) => !o)} className="text-gray-400 hover:text-white" title="Show flagged accounts">
          <ChevronDown size={14} />
        </button>
      </div>

      {open && (
        <div className="absolute z-20 mt-1 w-full rounded-lg border border-white/15 bg-neutral-950/95 shadow-xl">
          {!query && (
            <div className="flex border-b border-white/10 text-[11px]">
              {['fraud', 'mule'].map((g) => (
                <button
                  key={g}
                  type="button"
                  onClick={() => setTab(g)}
                  className={`flex-1 px-3 py-2 ${tab === g ? 'text-white border-b-2 border-cyan-400' : 'text-gray-500 hover:text-gray-300'}`}
                >
                  {labels[g]}
                </button>
              ))}
            </div>
          )}
          <div className="max-h-64 overflow-y-auto py-1 [color-scheme:dark]">
            {loading ? (
              <div className="flex items-center gap-2 px-3 py-3 text-xs text-gray-400"><Loader2 size={14} className="animate-spin" /> Loading…</div>
            ) : items.length === 0 ? (
              <div className="px-3 py-3 text-xs text-gray-500">{query ? 'No matching accounts.' : 'No accounts in this group.'}</div>
            ) : (
              items.map((a) => (
                <button
                  key={a.id}
                  type="button"
                  onClick={() => choose(a.id)}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-xs text-gray-200 hover:bg-white/10"
                >
                  <span className={`w-2 h-2 rounded-full shrink-0 ${GROUP_STYLE[a.group]?.dot}`} />
                  <span className="truncate">{a.id}</span>
                  <span className="ml-auto text-[10px] text-gray-500 shrink-0">{labels[a.group]}</span>
                </button>
              ))
            )}
          </div>
          {!loading && total > items.length && (
            <div className="border-t border-white/10 px-3 py-1.5 text-[10px] text-gray-500">
              Showing {items.length.toLocaleString()} of {total.toLocaleString()} — type to search.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function AiSection({ analysisId, account, aiEnabled }) {
  const [answers, setAnswers] = useState([])   // [{question, text, error}]
  const [loading, setLoading] = useState(false)
  const [question, setQuestion] = useState('')

  useEffect(() => { setAnswers([]); setQuestion('') }, [account])

  if (!aiEnabled) {
    return (
      <div className="flex gap-2 text-[11px] text-gray-500 border border-dashed border-white/10 rounded-lg p-3">
        <KeyRound size={14} className="shrink-0 mt-0.5" />
        <span>
          Optional: add <code className="text-gray-300">ANTHROPIC_API_KEY=…</code> to a <code className="text-gray-300">.env</code> file
          in the project folder and restart the backend to get an AI-written summary and ask follow-up questions.
          The evidence above works without it.
        </span>
      </div>
    )
  }

  const ask = async (q) => {
    setLoading(true)
    try {
      const data = await getJson(`${API}/api/analysis/${analysisId}/ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ account, question: q || null }),
      })
      setAnswers((a) => [...a, { question: q, text: data.ok ? data.text : null, error: data.ok ? null : data.error }])
      setQuestion('')
    } catch (err) {
      setAnswers((a) => [...a, { question: q, error: err.message }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-3">
      {answers.length === 0 && (
        <button
          type="button"
          disabled={loading}
          onClick={() => ask('')}
          className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-violet-500/15 border border-violet-400/40 text-violet-200 text-xs font-medium hover:bg-violet-500/25 transition-all disabled:opacity-50"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
          {loading ? 'Writing explanation…' : 'Explain with AI'}
        </button>
      )}

      {answers.map((a, i) => (
        <div key={i} className="rounded-lg border border-violet-400/25 bg-violet-500/5 p-3 text-xs text-gray-300 leading-relaxed space-y-2">
          {a.question && <p className="text-violet-200 font-medium">Q: {a.question}</p>}
          {a.error ? <p className="text-red-300">⚠ {a.error}</p> : <RichText text={a.text} />}
        </div>
      ))}

      {answers.length > 0 && (
        <form
          onSubmit={(e) => { e.preventDefault(); if (question.trim()) ask(question.trim()) }}
          className="flex items-center gap-2 bg-black/50 border border-white/15 rounded-lg px-3 py-2 focus-within:border-violet-400/60"
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a follow-up about this account…"
            className="flex-1 min-w-0 bg-transparent text-xs text-gray-200 placeholder:text-gray-500 outline-none"
          />
          <button type="submit" disabled={loading || !question.trim()} className="text-violet-300 disabled:opacity-40">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          </button>
        </form>
      )}
      <p className="text-[10px] text-gray-600">AI summaries are generated by Claude from the evidence above and can make mistakes.</p>
    </div>
  )
}

export default function AccountInvestigator({ analysisId, aiEnabled, labels, selected, onSelect }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!selected || !analysisId) { setData(null); setError(null); return }
    let cancelled = false
    setLoading(true)
    setError(null)
    getJson(`${API}/api/analysis/${analysisId}/explain?${new URLSearchParams({ account: selected })}`)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && (setError(e.message), setData(null)))
      .finally(() => !cancelled && setLoading(false))
    return () => { cancelled = true }
  }, [selected, analysisId])

  const style = GROUP_STYLE[data?.group] ?? GROUP_STYLE.normal

  return (
    <div className="flex flex-col gap-4 min-h-0">
      <div>
        <h2 className="text-xl font-medium mb-1">3. Account Investigator</h2>
        <p className="text-gray-400 text-xs">Why is an account flagged? Click a node, search, or pick from the list.</p>
      </div>

      <AccountPicker analysisId={analysisId} labels={labels} onSelect={onSelect} />

      <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-4 [color-scheme:dark]">
        {!selected && (
          <div className="text-center text-xs text-gray-500 py-10 space-y-2">
            <Search size={20} className="mx-auto text-gray-600" />
            <p>No account selected yet.</p>
          </div>
        )}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-gray-400 py-6 justify-center"><Loader2 size={16} className="animate-spin" /> Gathering evidence…</div>
        )}
        {error && <p className="text-xs text-red-300">⚠ {error}</p>}

        {data && !loading && (
          <>
            <div className="space-y-2">
              <div className="flex items-start justify-between gap-2">
                <p className="font-mono text-sm text-white break-all">{data.account}</p>
                <div className="flex items-center gap-1 shrink-0">
                  <span className={`text-[11px] border px-2 py-0.5 rounded-full ${style.badge}`}>{labels[data.group]}</span>
                  <button type="button" onClick={() => onSelect(null)} className="p-1 text-gray-500 hover:text-white" title="Clear">
                    <X size={14} />
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-300 leading-relaxed">{data.summary}</p>
            </div>

            {data.reasons.length > 0 && (
              <div className="space-y-2">
                <p className="text-[11px] uppercase tracking-wider text-gray-500">Evidence</p>
                <ul className="space-y-2">
                  {data.reasons.map((r, i) => (
                    <li key={i} className="flex gap-2 text-xs">
                      {KIND_ICON[r.kind] ?? KIND_ICON.info}
                      <div>
                        <p className="text-gray-100 font-medium">{r.title}</p>
                        <p className="text-gray-400 leading-relaxed">{r.detail}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              {data.stats.map((s) => (
                <div key={s.label} className="rounded-lg bg-black/40 border border-white/5 px-2.5 py-2">
                  <p className="text-[10px] text-gray-500">{s.label}</p>
                  <p className="text-xs text-gray-200">{s.value}</p>
                </div>
              ))}
            </div>

            {data.unusual_features?.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[11px] uppercase tracking-wider text-gray-500">Most unusual values</p>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-[10px] text-gray-500 text-left"><th className="font-normal">Feature</th><th className="font-normal">Value</th><th className="font-normal">Typical</th></tr>
                  </thead>
                  <tbody>
                    {data.unusual_features.map((f) => (
                      <tr key={f.feature} className="text-gray-300">
                        <td className="py-0.5 pr-2 truncate">{f.feature}</td>
                        <td className={f.direction === 'higher' ? 'text-amber-300' : 'text-sky-300'}>{f.value} {f.direction === 'higher' ? '↑' : '↓'}</td>
                        <td className="text-gray-500">{f.typical}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {data.counterparties.length > 0 && (
              <div className="space-y-1.5">
                <p className="text-[11px] uppercase tracking-wider text-gray-500">
                  {data.mode === 'transactions' ? 'Top counterparties' : 'Most similar customers'}
                </p>
                <ul className="space-y-1">
                  {data.counterparties.map((c, i) => (
                    <li key={`${c.id}-${c.relation}-${i}`}>
                      <button
                        type="button"
                        onClick={() => onSelect(c.id)}
                        className="w-full flex items-center gap-2 text-xs text-left px-2 py-1 rounded hover:bg-white/10 group"
                        title={`Investigate ${c.id}`}
                      >
                        <span className={`w-2 h-2 rounded-full shrink-0 ${GROUP_STYLE[c.group]?.dot}`} />
                        <span className="text-gray-500 shrink-0 w-24">{c.relation}</span>
                        <span className="truncate text-gray-200">{c.id}</span>
                        <span className="ml-auto text-[10px] text-gray-500 shrink-0">
                          {c.transactions != null && `${c.transactions}×`}{c.amount != null && ` · ${c.amount.toLocaleString()}`}
                        </span>
                        <ArrowRight size={12} className="text-gray-600 group-hover:text-white shrink-0" />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <AiSection analysisId={analysisId} account={data.account} aiEnabled={aiEnabled} />
          </>
        )}
      </div>
    </div>
  )
}
