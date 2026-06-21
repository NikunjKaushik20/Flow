import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { Share2, Maximize2, Minimize2, Layers, Brain } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';

const COLORS = {
  critical: '#FF5D5D',
  moderate: '#FFB84D',
  minor: '#47D18C',
  bg: '#13141C',
  bgLight: '#1a1a2e',
  edge: 'rgba(158, 193, 255, 0.3)',
  edgeHighlight: 'rgba(158, 193, 255, 0.7)',
  text: '#F5F7FA',
  textMuted: '#A7B0BE',
};

const NetworkGraph = ({ data }) => {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoverNode, setHoverNode] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [viewMode, setViewMode] = useState('risk'); // 'risk' or 'topology'

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: Math.max(500, containerRef.current.clientHeight - 80),
        });
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [isFullscreen]);

  const graphData = useMemo(() => {
    if (!data?.corridors?.length) return { nodes: [], links: [] };
    
    const nodeIds = new Set(data.corridors.map(c => c.name));
    
    const nodes = data.corridors.map(c => ({
      id: c.name,
      name: c.name,
      val: Math.max(8, Math.min(25, (c.incident_count || 10) / 8)),
      risk: c.critical_rate > 0.15 ? 'High' : c.critical_rate > 0.08 ? 'Moderate' : 'Low',
      color: c.critical_rate > 0.15 ? COLORS.critical : c.critical_rate > 0.08 ? COLORS.moderate : COLORS.minor,
      incidents: c.incident_count,
      rate: c.critical_rate,
      x: (Math.random() - 0.5) * 400,
      y: (Math.random() - 0.5) * 400,
    }));

    // Filter links to only include edges where both source and target exist in nodes
    const links = (data.edges || [])
      .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map(e => ({
        source: e.source,
        target: e.target,
        value: e.cooccurrence || e.weight || 1,
        curvature: 0.1,
        spillover: e.spillover !== undefined ? e.spillover : 0.5,
      }));

    return { nodes, links };
  }, [data]);
  
  // Color edges by spillover in topology view
  const getLinkColor = useCallback((link) => {
    if (viewMode === 'topology') {
      const spillover = link.spillover || 0.5;
      // Green = low spillover (good diversion), Red = high spillover (bad diversion)
      const g = Math.round(255 * (1 - spillover));
      const r = Math.round(255 * spillover);
      return `rgba(${r}, ${g}, 50, 0.6)`;
    }
    return (hoverNode?.id === link.source.id || hoverNode?.id === link.target.id || 
            selectedNode?.id === link.source.id || selectedNode?.id === link.target.id) 
      ? COLORS.edgeHighlight : COLORS.edge;
  }, [hoverNode, selectedNode, viewMode]);
  
  const getLinkWidth = useCallback((link) => {
    if (viewMode === 'topology') {
      const spillover = link.spillover || 0.5;
      // Thicker = lower spillover (better diversion route)
      return Math.max(1, 4 * (1 - spillover));
    }
    return Math.max(1, link.value * 2);
  }, [viewMode]);

  const paintNode = useCallback((node, ctx, globalScale) => {
    const size = node.val;
    const isHovered = hoverNode?.id === node.id;
    const isSelected = selectedNode?.id === node.id;
    
    // Node glow effect
    if (isHovered || isSelected) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size + 4, 0, 2 * Math.PI);
      ctx.fillStyle = node.color + '40'; // 25% opacity
      ctx.fill();
    }
    
    // Main node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = node.color;
    ctx.fill();
    
    // Inner highlight
    ctx.beginPath();
    ctx.arc(node.x - size * 0.3, node.y - size * 0.3, size * 0.3, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.fill();
    
    // Border
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.strokeStyle = isHovered || isSelected ? '#fff' : 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = isHovered || isSelected ? 2 : 1;
    ctx.stroke();
    
    // Label
    const fontSize = Math.max(10, 14 / globalScale);
    ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Text shadow for readability
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillText(node.name, node.x, node.y + size + fontSize + 4);
    ctx.fillStyle = COLORS.text;
    ctx.fillText(node.name, node.x, node.y + size + fontSize + 3);
  }, [hoverNode, selectedNode]);

  const handleClick = useCallback((node) => {
    setSelectedNode(selectedNode?.id === node.id ? null : node);
  }, [selectedNode]);

  if (!graphData.nodes.length) {
    return (
      <div className="panel" style={{ padding: 'var(--space-5)', textAlign: 'center' }}>
        <Share2 size={48} className="text-muted" />
        <h2 className="text-md" style={{ marginTop: 'var(--space-3)' }}>Loading Network Graph...</h2>
        <p className="text-muted text-sm">Connect to the API to visualize the corridor network</p>
      </div>
    );
  }

  // Calculate stats for the info panel
  const highRiskCount = graphData.nodes.filter(n => n.risk === 'High').length;
  const moderateRiskCount = graphData.nodes.filter(n => n.risk === 'Moderate').length;
  const lowRiskCount = graphData.nodes.filter(n => n.risk === 'Low').length;
  const totalIncidents = graphData.nodes.reduce((sum, n) => sum + (n.incidents || 0), 0);

  return (
    <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }} ref={containerRef}>
      {/* Header */}
      <div className="section-heading" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <Share2 size={18} />
          <h2>Corridor Network Topology</h2>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
          <span className="text-xs text-muted">
            {graphData.nodes.length} corridors • {graphData.links.length} connections
          </span>
          <button 
            onClick={() => setIsFullscreen(!isFullscreen)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: COLORS.textMuted }}
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>
 
      {/* View Mode Toggle & Legend */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-3)', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
        <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
          <button 
            onClick={() => setViewMode('risk')}
            style={{ 
              background: viewMode === 'risk' ? 'rgba(71, 209, 140, 0.2)' : 'transparent',
              border: viewMode === 'risk' ? '1px solid #47D18C' : '1px solid var(--border-subtle)',
              color: viewMode === 'risk' ? '#47D18C' : COLORS.textMuted,
              padding: '4px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, cursor: 'pointer'
            }}
          >
            <Layers size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Risk View
          </button>
          <button 
            onClick={() => setViewMode('topology')}
            style={{ 
              background: viewMode === 'topology' ? 'rgba(158, 193, 255, 0.2)' : 'transparent',
              border: viewMode === 'topology' ? '1px solid #9EC1FF' : '1px solid var(--border-subtle)',
              color: viewMode === 'topology' ? '#9EC1FF' : COLORS.textMuted,
              padding: '4px 10px', borderRadius: 'var(--radius-sm)', fontSize: 12, cursor: 'pointer'
            }}
          >
            <Brain size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} /> Learned Topology
          </button>
        </div>
        <div style={{ display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
          {viewMode === 'risk' && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                <span style={{ width: 12, height: 12, borderRadius: '50%', background: COLORS.critical }}></span>
                <span className="text-xs">High Risk ({highRiskCount})</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                <span style={{ width: 12, height: 12, borderRadius: '50%', background: COLORS.moderate }}></span>
                <span className="text-xs">Moderate ({moderateRiskCount})</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-1)' }}>
                <span style={{ width: 12, height: 12, borderRadius: '50%', background: COLORS.minor }}></span>
                <span className="text-xs">Low Risk ({lowRiskCount})</span>
              </div>
            </>
          )}
          {viewMode === 'topology' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <span className="text-xs text-muted">Green = Low Spillover (Good Diversion)</span>
              <span style={{ width: 20, height: 12, borderRadius: '2px', background: 'linear-gradient(90deg, #FF5D5D, #FFB84D, #47D18C)' }}></span>
              <span className="text-xs text-muted">Red = High Spillover (Bad Diversion)</span>
            </div>
          )}
        </div>
      </div>
 
      {/* Graph Container */}
      <div style={{ 
        flex: 1, 
        position: 'relative', 
        borderRadius: 'var(--radius-md)', 
        overflow: 'hidden', 
        border: '1px solid var(--border-subtle)',
        background: COLORS.bg,
      }}>
        <ForceGraph2D
          width={dimensions.width - 40}
          height={dimensions.height - 40}
          graphData={graphData}
          nodeLabel=""
          nodeColor={n => n.color}
          nodeVal={n => n.val}
          nodeRelSize={1}
          nodeCanvasObject={paintNode}
          linkColor={getLinkColor}
          linkWidth={getLinkWidth}
          linkCurvature="curvature"
          linkDirectionalParticles={2}
          linkDirectionalParticleSpeed={0.005}
          linkDirectionalParticleColor={() => COLORS.edgeHighlight}
          backgroundColor={COLORS.bg}
          onNodeHover={node => setHoverNode(node)}
          onNodeClick={handleClick}
          cooldownTicks={100}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          enableNodeDrag={true}
          enableZoomInteraction={true}
          enablePanInteraction={true}
        />
        
        {/* Hover Tooltip */}
        {hoverNode && (
          <div style={{
            position: 'absolute',
            bottom: 'var(--space-3)',
            left: 'var(--space-3)',
            background: COLORS.bgLight,
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: 'var(--space-3)',
            minWidth: 180,
          }}>
            <div style={{ fontWeight: 600, marginBottom: 'var(--space-2)' }}>{hoverNode.name}</div>
            <div style={{ display: 'grid', gap: 'var(--space-1)', fontSize: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.textMuted }}>Risk Level</span>
                <span style={{ color: hoverNode.color, fontWeight: 600 }}>{hoverNode.risk}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.textMuted }}>Incidents</span>
                <span>{hoverNode.incidents || '—'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: COLORS.textMuted }}>Critical Rate</span>
                <span>{((hoverNode.rate || 0) * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NetworkGraph;
