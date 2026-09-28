import { useRef, useEffect, useMemo, useState, useCallback } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Maximize2 } from 'lucide-react'

/**
 * FraudNetworkGraph.jsx
 * Renders the graph_data payload returned by POST /api/analyze.
 *
 * Props
 * -----
 * graphData : { nodes: [{id, group}], links: [{source, target}] }
 * height    : canvas height (default 480). Width always follows the container.
 * labels    : legend / tooltip text per group (default: transaction wording)
 * selectedId: account to highlight and fly to (optional)
 * onNodeClick(id): called when a node is clicked (optional)
 */

const GROUP_COLOR = {
  fraud:  '#ef4444',   // red
  mule:   '#22d3ee',   // cyan
  normal: '#6b7280',   // gray
}

// Translucent halo colours — replaces canvas shadowBlur, which is very slow per node
const HALO_COLOR = {
  fraud:  'rgba(239, 68, 68, 0.25)',
  mule:   'rgba(34, 211, 238, 0.28)',
  normal: 'rgba(107, 114, 128, 0.18)',
}

const GROUP_LABEL = { fraud: 'Fraud actor', mule: 'Mule / offshore', normal: 'Normal' }

const endpointId = (end) => (typeof end === 'object' ? end.id : end)

export default function FraudNetworkGraph({ graphData, height = 480, labels = GROUP_LABEL, selectedId = null, onNodeClick }) {
  const graphRef     = useRef()
  const containerRef = useRef()
  const fittedRef    = useRef(false)
  const nodeCache    = useRef(new Map())   // id -> node object, so positions survive graph additions
  const [width, setWidth] = useState(0)

  // Follow the container's width so the graph is never cropped or off-centre
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Build the graph object when the data changes (not on every render). Node objects are
  // reused so that accounts added later (from the investigator) don't reshuffle the layout.
  const enriched = useMemo(() => {
    if (!graphData?.nodes?.length) return null
    const cache = nodeCache.current
    const nodes = graphData.nodes.map(n => {
      let node = cache.get(n.id)
      if (!node) {
        node = { ...n, color: GROUP_COLOR[n.group] ?? '#ffffff' }
        cache.set(n.id, node)
      }
      return node
    })
    const links = graphData.links.map(l => ({ source: endpointId(l.source), target: endpointId(l.target) }))
    // Drop newly added nodes next to an already-placed neighbour instead of at the origin
    for (const l of links) {
      const [a, b] = [cache.get(l.source), cache.get(l.target)]
      for (const [fresh, anchor] of [[a, b], [b, a]]) {
        if (fresh && anchor && fresh.x == null && anchor.x != null) {
          fresh.x = anchor.x + (Math.random() - 0.5) * 30
          fresh.y = anchor.y + (Math.random() - 0.5) * 30
        }
      }
    }
    return { nodes, links }
  }, [graphData])

  const ready = width > 0
  useEffect(() => {
    // Fresh simulation for every dataset (the component is re-mounted per dataset,
    // which prevents grey overlapped tangles on CSV replace); gentle reheat on additions
    const fg = graphRef.current
    if (!fg || !enriched) return
    fg.d3Force('charge').strength(-180)
    fg.d3Force('link').distance(50)
    fg.d3ReheatSimulation()
  }, [enriched, ready])

  const fitView = useCallback(() => graphRef.current?.zoomToFit(400, 40), [])

  // Fly to the selected account (e.g. picked from search). Newly added nodes need a
  // few simulation ticks before they have a position, so retry briefly.
  useEffect(() => {
    if (!selectedId || !enriched) return
    let tries = 0
    let timer
    const fly = () => {
      const fg = graphRef.current
      const node = enriched.nodes.find(n => n.id === selectedId)
      if (fg && node && node.x != null) {
        fittedRef.current = true   // the user is focused on an account — don't auto-fit away from it
        fg.centerAt(node.x, node.y, 600)
        if (fg.zoom() < 2.5) fg.zoom(2.5, 600)
        return
      }
      if (++tries < 15) timer = setTimeout(fly, 150)
    }
    fly()
    return () => clearTimeout(timer)
  }, [selectedId, enriched])

  const isSelectedLink = useCallback(
    (l) => selectedId != null && (endpointId(l.source) === selectedId || endpointId(l.target) === selectedId),
    [selectedId],
  )

  // Auto-fit only the first time the layout settles, not after every node drag
  const handleEngineStop = useCallback(() => {
    if (fittedRef.current) return
    fittedRef.current = true
    fitView()
  }, [fitView])

  const drawNode = useCallback((node, ctx, globalScale) => {
    const r = node.group === 'mule' ? 7 : 5

    // glow
    ctx.beginPath()
    ctx.arc(node.x, node.y, r * (node.group === 'mule' ? 2 : 1.7), 0, 2 * Math.PI)
    ctx.fillStyle = HALO_COLOR[node.group] ?? 'rgba(255,255,255,0.15)'
    ctx.fill()

    ctx.beginPath()
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
    ctx.fillStyle = node.color
    ctx.fill()

    // selection ring
    const selected = node.id === selectedId
    if (selected) {
      ctx.beginPath()
      ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI)
      ctx.strokeStyle = '#facc15'
      ctx.lineWidth = 2.5 / Math.max(globalScale, 0.5)
      ctx.stroke()
    }

    // label
    if (globalScale > 1.2 || selected) {
      ctx.font      = `${Math.max(10 / globalScale, 3)}px sans-serif`
      ctx.fillStyle = '#e5e7eb'
      ctx.textAlign = 'center'
      ctx.fillText(node.id, node.x, node.y - r - 2)
    }
  }, [selectedId])

  const drawPointerArea = useCallback((node, color, ctx) => {
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(node.x, node.y, node.group === 'mule' ? 9 : 7, 0, 2 * Math.PI)
    ctx.fill()
  }, [])

  if (!enriched) return null

  return (
    <div ref={containerRef} className="relative w-full" style={{ height }}>
      {ready && (
        <ForceGraph2D
          ref={graphRef}
          graphData={enriched}
          width={width}
          height={height}
          backgroundColor="rgba(0,0,0,0)"
          nodeLabel={n => `${n.id} · ${labels[n.group] ?? n.group}`}
          nodeRelSize={6}
          nodeColor={n => n.color}
          linkColor={l => (isSelectedLink(l) ? 'rgba(250,204,21,0.8)' : 'rgba(255,255,255,0.12)')}
          linkWidth={l => (isSelectedLink(l) ? 2 : 1)}
          onNodeClick={n => onNodeClick?.(n.id)}
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={drawPointerArea}
          cooldownTicks={100}
          onEngineStop={handleEngineStop}
        />
      )}

      {/* Legend */}
      <div className="absolute left-3 bottom-3 flex flex-wrap gap-3 text-[11px] text-gray-300 bg-black/60 border border-white/10 rounded-lg px-3 py-2 pointer-events-none">
        {Object.entries(labels).map(([g, label]) => (
          <span key={g} className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: GROUP_COLOR[g] }} />
            {label}
          </span>
        ))}
      </div>

      <button
        type="button"
        onClick={fitView}
        className="absolute right-3 top-3 inline-flex items-center gap-1.5 text-[11px] text-gray-300 bg-black/60 border border-white/10 rounded-lg px-3 py-1.5 hover:text-white hover:border-white/30 transition-colors"
        title="Zoom to fit the whole network"
      >
        <Maximize2 size={12} /> Fit to view
      </button>
    </div>
  )
}
