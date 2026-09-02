import { useRef, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

/**
 * FraudNetworkGraph.jsx
 * Renders the graph_data payload returned by POST /api/analyze.
 *
 * Props
 * -----
 * graphData : { nodes: [{id, group}], links: [{source, target}] }
 * width     : canvas width  (default 100% of container)
 * height    : canvas height (default 420)
 */

const GROUP_COLOR = {
  fraud:  '#ef4444',   // red
  mule:   '#22d3ee',   // cyan
  normal: '#6b7280',   // gray
}

export default function FraudNetworkGraph({ graphData, width, height = 420 }) {
  const graphRef = useRef()

  useEffect(() => {
    // Force a fresh simulation on every new dataset — prevents grey overlapped tangles
    // when replacing one CSV with another (old node positions/links retained otherwise)
    if (graphRef.current) {
      const fg = graphRef.current
      fg.d3Force('charge').strength(-180)
      fg.d3Force('link').distance(50)
      fg.d3ReheatSimulation()
      // Re-center after layout stabilises
      const t = setTimeout(() => fg.zoomToFit(400, 40), 600)
      return () => clearTimeout(t)
    }
  }, [graphData])

  if (!graphData || !graphData.nodes?.length) return null

  // Enrich nodes with colour
  const enriched = {
    nodes: graphData.nodes.map(n => ({
      ...n,
      color: GROUP_COLOR[n.group] ?? '#ffffff',
    })),
    links: graphData.links,
  }

  return (
    <ForceGraph2D
      ref={graphRef}
      graphData={enriched}
      width={width}
      height={height}
      backgroundColor="transparent"
      nodeLabel="id"
      nodeRelSize={6}
      nodeColor={n => n.color}
      linkColor={() => 'rgba(255,255,255,0.12)'}
      linkWidth={1}
      nodeCanvasObject={(node, ctx, globalScale) => {
        const r    = node.group === 'mule' ? 7 : 5
        const font = Math.max(10 / globalScale, 3)

        // glow
        ctx.shadowBlur  = node.group === 'mule' ? 14 : 8
        ctx.shadowColor = node.color

        ctx.beginPath()
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
        ctx.fillStyle = node.color
        ctx.fill()
        ctx.shadowBlur = 0

        // label
        if (globalScale > 1.2) {
          ctx.font      = `${font}px sans-serif`
          ctx.fillStyle = '#e5e7eb'
          ctx.textAlign = 'center'
          ctx.fillText(node.id, node.x, node.y - r - 2)
        }
      }}
      cooldownTicks={100}
      onEngineStop={() => graphRef.current?.zoomToFit(400, 40)}
    />
  )
}
