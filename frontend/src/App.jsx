import React, { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, Activity, Clock, FileText, Map as MapIcon, MapPin, Navigation, ShieldCheck, X, BarChart3, Cpu, Calendar, Zap } from 'lucide-react';
import { CircleMarker, MapContainer, Popup, TileLayer, useMap, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './index.css';
import './App.css';
import ScenarioForm from './components/ScenarioForm';
import IncidentDrawer from './components/IncidentDrawer';
import TopBar from './components/TopBar';
import Sidebar from './components/Sidebar';
import AnalyticsPanel from './components/AnalyticsPanel';
import ResourcePanel from './components/ResourcePanel';
import ModelIntelligence from './components/ModelIntelligence';
import NetworkGraph from './components/NetworkGraph';

// Use relative URL when proxy is configured, absolute URL for direct access
const API = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? 'http://localhost:8000' : '');

const MapUpdater = ({ center, zoom }) => {
  const map = useMap();
  React.useEffect(() => {
    if (center && center[0] != null && center[1] != null) {
      map.flyTo(center, zoom || map.getZoom());
    }
  }, [center, zoom, map]);
  return null;
};

function App() {
  const [activeTab, setActiveTab] = useState('command');
  const [predictionData, setPredictionData] = useState(null);
  const [simulationData, setSimulationData] = useState(null);
  const [optimizationData, setOptimizationData] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [corridors, setCorridors] = useState([]);
  const [graphData, setGraphData] = useState({ corridors: [], edges: [] });
  const [historyData, setHistoryData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [analyticsData, setAnalyticsData] = useState(null);

  // Fetch real data on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [corrRes, graphRes, histRes, metRes, analyticsRes] = await Promise.all([
          fetch(`${API}/api/corridors`).catch(() => null),
          fetch(`${API}/api/graph`).catch(() => null),
          fetch(`${API}/api/history`).catch(() => null),
          fetch(`${API}/api/metrics`).catch(() => null),
          fetch(`${API}/api/analytics`).catch(() => null),
        ]);
        if (corrRes?.ok) {
          const d = await corrRes.json();
          // Handle both array response and object with corridors property
          setCorridors(Array.isArray(d) ? d : (d.corridors || []));
        }
        if (graphRes?.ok) setGraphData(await graphRes.json());
        if (histRes?.ok) setHistoryData(await histRes.json());
        if (metRes?.ok) setMetricsData(await metRes.json());
        if (analyticsRes?.ok) setAnalyticsData(await analyticsRes.json());
      } catch (e) {
        console.warn('API not reachable, using fallback data');
      }
    };
    fetchData();
  }, []);

  const handlePredict = async (formData) => {
    setLoading(true);
    setError(null);
    setIsDrawerOpen(false);
    setSimulationData(null);
    setOptimizationData(null);
    try {
      // 1. Predict
      const predRes = await fetch(`${API}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!predRes.ok) throw new Error('Prediction failed');
      const predData = await predRes.json();

      // 2. Simulate
      const simRes = await fetch(`${API}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          corridor: formData.corridor,
          severity: predData.severity,
          duration_minutes: predData.pred_duration_bucket === '2hr+' ? 150 :
            predData.pred_duration_bucket === '30min-2hr' ? 75 : 20,
          hour_of_day: new Date().getHours(),
        }),
      });
      const simData = simRes.ok ? await simRes.json() : null;

      // 3. Optimize
      let optData = null;
      if (simData?.affected_corridors) {
        const optRes = await fetch(`${API}/api/optimize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            affected_corridors: simData.affected_corridors,
            total_officers: 20,
            total_barricades: 10,
          }),
        });
        optData = optRes.ok ? await optRes.json() : null;
      }

      const selectedCorridor = corridors.find(c => c.name === formData.corridor) ||
        graphData.corridors?.find(c => c.name === formData.corridor);
      const center = selectedCorridor ? [selectedCorridor.lat, selectedCorridor.lon || selectedCorridor.lng] : [12.9716, 77.5946];

      setPredictionData({
        ...predData,
        center,
        event_title: `${formData.event_cause || 'Incident'} on ${formData.corridor || 'Unknown corridor'}`,
      });
      setSimulationData(simData);
      setOptimizationData(optData);
      setIsDrawerOpen(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getDelayColor = (delay) => {
    if (delay >= 3.0) return '#FF5D5D';
    if (delay >= 1.5) return '#FFB84D';
    if (delay >= 1.1) return '#63A4FF';
    return '#47D18C';
  };

  const getRiskColor = (risk) => {
    if (risk === 'High') return '#FF5D5D';
    if (risk === 'Moderate') return '#FFB84D';
    return '#47D18C';
  };

  const mapCorridors = corridors.length > 0 ? corridors : (graphData.corridors || []).map(c => ({
    ...c, risk: c.critical_rate > 0.15 ? 'High' : c.critical_rate > 0.08 ? 'Moderate' : 'Low'
  }));

  // ============================================================
  // COMMAND CENTER
  // ============================================================
  const renderCommandCenter = () => {
    const totalIncidents = historyData?.total_events || 0;
    const sevDist = historyData?.severity_distribution || {};
    const critRate = totalIncidents > 0 ? ((sevDist['2hr+'] || 0) / totalIncidents * 100).toFixed(1) : '0';
    const modelAccuracy = metricsData?.model?.accuracy ? (metricsData.model.accuracy * 100).toFixed(1) : '--';
    const macroF1 = metricsData?.model?.macro_f1?.toFixed(3) || '--';
    const critRecall = metricsData?.model?.critical_recall ? (metricsData.model.critical_recall * 100).toFixed(1) : metricsData?.model?.['2hr+']?.recall ? (metricsData.model['2hr+'].recall * 100).toFixed(1) : '--';

    return (
      <div className="dashboard-grid">
        <section className="stat-strip">
          <div className="stat-tile">
            <span>Total Events</span>
            <strong>{totalIncidents.toLocaleString()}</strong>
            <small>Historical incidents</small>
          </div>
          <div className="stat-tile">
            <span>Critical Rate</span>
            <strong>{critRate}%</strong>
            <small>2hr+ incidents</small>
          </div>
          <div className="stat-tile">
            <span>Critical Recall</span>
            <strong>{critRecall}%</strong>
            <small>2hr+ incidents caught</small>
          </div>
          <div className="stat-tile">
            <span>Model</span>
            <strong>{metricsData?.model?.best_approach === 'Tree 80% + GNN 15% + AutoGluon 5%' ? '80+15+5 Blend' : 'Ensemble'}</strong>
            <small>{metricsData?.model?.selected_models ? metricsData.model.selected_models.join(' + ') : 'LightGBM + AdaBoost + CatBoost'}</small>
          </div>
        </section>

        <section className="ops-layout">
          <div className="panel live-map-panel">
            <div className="section-heading">
              <MapIcon size={18} />
              <h2>Network State</h2>
            </div>
            <MapContainer center={[12.9716, 77.5946]} zoom={11} scrollWheelZoom className="leaflet-map">
              <TileLayer
                attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              />
              {mapCorridors.filter(c => c.name !== 'Non-corridor').map(corridor => (
                <CircleMarker
                  key={corridor.name}
                  center={[corridor.lat, corridor.lon || corridor.lng || 77.59]}
                  radius={Math.max(8, Math.min(20, (corridor.incident_count || 30) / 10))}
                  pathOptions={{
                    color: getRiskColor(corridor.risk),
                    fillColor: getRiskColor(corridor.risk),
                    fillOpacity: 0.7,
                    weight: 2,
                  }}
                >
                  <Popup>
                    <strong>{corridor.name}</strong><br />
                    Risk: {corridor.risk}<br />
                    Incidents: {corridor.incident_count || '—'}<br />
                    Critical rate: {((corridor.critical_rate || 0) * 100).toFixed(1)}%
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          <div className="panel">
            <div className="section-heading">
              <AlertTriangle size={18} />
              <h2>Quick Actions</h2>
            </div>
            <div className="prediction-summary">
              {predictionData ? (
                <>
                  <strong>{predictionData.event_title}</strong>
                  <span className={`status-pill ${predictionData.severity.toLowerCase()}`}>{predictionData.severity}</span>
                  <div className="summary-row">
                    <span>Duration</span><b>{predictionData.pred_duration_bucket}</b>
                  </div>
                  <div className="summary-row">
                    <span>Impact</span><b>{predictionData.impact_score} / 10</b>
                  </div>
                </>
              ) : (
                <>
                  <strong>No active prediction</strong>
                  <p className="text-muted text-sm">Run a simulation to see impact analysis</p>
                </>
              )}
              <button className="btn-accent" onClick={() => setActiveTab('simulate')}>
                <Zap size={14} style={{ marginRight: 6 }} />Run Simulation
              </button>
            </div>

            {simulationData && (
              <div style={{ marginTop: 'var(--space-3)' }}>
                <div className="section-heading">
                  <Activity size={18} />
                  <h2>Last Simulation</h2>
                </div>
                <div className="summary-row">
                  <span>Affected corridors</span>
                  <b>{simulationData.total_affected}</b>
                </div>
                <div className="summary-row">
                  <span>Vehicle-hours delay</span>
                  <b>{simulationData.total_delay_vehicle_hours?.toLocaleString()}</b>
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    );
  };

  // ============================================================
  // IMPACT SIMULATOR
  // ============================================================
  const renderSimulator = () => (
    <div className="scenario-view">
      <div className="form-container">
        <div className="section-heading">
          <Zap size={18} />
          <h2>Impact Simulator</h2>
        </div>
        <p className="text-muted text-sm" style={{ marginBottom: 'var(--space-3)' }}>
          Configure an event and see how congestion cascades through the road network.
        </p>
        {error && <div className="error-banner" style={{ marginBottom: 'var(--space-3)' }}>{error}</div>}
        <ScenarioForm onSubmit={handlePredict} loading={loading} corridors={mapCorridors} />
      </div>
      <div className="panel live-map-panel">
        <div className="section-heading">
          <MapIcon size={18} />
          <h2>Congestion Cascade</h2>
        </div>
        <MapContainer
          center={predictionData?.center || [12.9716, 77.5946]}
          zoom={11}
          scrollWheelZoom
          className="leaflet-map"
        >
          <MapUpdater center={predictionData?.center || [12.9716, 77.5946]} />
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          {mapCorridors.filter(c => c.name !== 'Non-corridor').map(corridor => {
            const simDelay = simulationData?.peak_delays?.[corridor.name] || 1.0;
            const color = simulationData ? getDelayColor(simDelay) : getRiskColor(corridor.risk);
            const radius = simulationData ? Math.max(8, Math.min(24, simDelay * 8)) : 12;

            return (
              <CircleMarker
                key={corridor.name}
                center={[corridor.lat, corridor.lon || corridor.lng || 77.59]}
                radius={radius}
                pathOptions={{
                  color: color,
                  fillColor: color,
                  fillOpacity: simulationData ? 0.85 : 0.6,
                  weight: simulationData ? 3 : 2,
                }}
              >
                <Popup>
                  <strong>{corridor.name}</strong><br />
                  {simulationData ? (
                    <>Delay: {simDelay}x<br />Status: {simDelay >= 3 ? 'CRITICAL' : simDelay >= 1.5 ? 'CONGESTED' : 'NORMAL'}</>
                  ) : (
                    <>Risk: {corridor.risk}</>
                  )}
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>

        {simulationData && (
          <div className="sim-cascade-strip">
            {simulationData.affected_corridors?.slice(0, 6).map(ac => (
              <div key={ac.corridor} className={`cascade-chip ${ac.is_event_corridor ? 'event' : ac.peak_delay >= 1.5 ? 'congested' : 'minor'}`}>
                <span className="cascade-name">{ac.corridor}</span>
                <span className="cascade-delay">{ac.peak_delay}x</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  // ============================================================
  // CORRIDORS
  // ============================================================
  const renderCorridors = () => (
    <div className="table-panel panel">
      <div className="section-heading">
        <MapPin size={18} />
        <h2>Corridor Intelligence</h2>
      </div>
      <div className="corridor-table">
        <div className="corridor-table-header">
          <span>Corridor</span>
          <span>Load</span>
          <span>Incidents</span>
          <span>Risk</span>
        </div>
        {mapCorridors.filter(c => c.name !== 'Non-corridor').map(corridor => (
          <div key={corridor.name} className="corridor-table-row">
            <span>{corridor.name}</span>
            <div className="load-track">
              <i style={{ width: `${Math.min(100, (corridor.critical_rate || 0) * 500)}%`, background: getRiskColor(corridor.risk) }}></i>
            </div>
            <b>{corridor.incident_count || '—'}</b>
            <em className={`risk-text ${(corridor.risk || 'low').toLowerCase()}`}>{corridor.risk}</em>
          </div>
        ))}
      </div>
    </div>
  );

  // ============================================================
  // RULES
  // ============================================================
  const renderRules = () => (
    <div className="rules-panel panel">
      <div className="section-heading">
        <ShieldCheck size={18} />
        <h2>Operational Protocol</h2>
      </div>
      <div className="rule-list">
        <div className="rule-card critical-border">
          <b className="severity-critical">CRITICAL</b>
          <span>Dispatch 4 officers + SI. Full lane barricading. Activate diversion protocol. Response: 5 min.</span>
        </div>
        <div className="rule-card moderate-border">
          <b className="severity-moderate">MODERATE</b>
          <span>Dispatch 2 patrol officers. Traffic cones + warning signs. Prepare diversion. Response: 15 min.</span>
        </div>
        <div className="rule-card minor-border">
          <b className="severity-minor">MINOR</b>
          <span>Dispatch 1 patrol bike. Monitor corridor flow. Response: 30 min.</span>
        </div>
      </div>
      <div style={{ marginTop: 'var(--space-4)' }}>
        <div className="section-heading">
          <Activity size={18} />
          <h2>Layer Architecture</h2>
        </div>
        <div className="rule-list">
          <div><b>Layer 0</b><span>Road Network Graph — 22 corridors, 294 junctions, 48 inter-corridor edges</span></div>
          <div><b>Layer 1</b><span>Rule-Based Triage — High priority + road closure = instant CRITICAL flag</span></div>
          <div><b>Layer 2</b><span>Ensemble ML — LightGBM + AdaBoost + CatBoost duration prediction</span></div>
          <div><b>Layer 3</b><span>Congestion Simulator — BPR-based cascade propagation across network</span></div>
          <div><b>Layer 4</b><span>Resource Optimizer — Greedy allocation minimizing vehicle-delay-hours</span></div>
          <div><b>Layer 5</b><span>Post-Event Learning — Calibration tracking + per-corridor bias correction</span></div>
        </div>
      </div>
    </div>
  );

  // ============================================================
  // TAB ROUTER
  // ============================================================
  const renderActiveTab = () => {
    switch (activeTab) {
      case 'command': return renderCommandCenter();
      case 'simulate': return renderSimulator();
      case 'optimizer': return <ResourcePanel simulationData={simulationData} />;
      case 'analytics': return <AnalyticsPanel data={analyticsData} />;
      case 'network': return <NetworkGraph data={graphData} />;
      case 'corridors': return renderCorridors();
      case 'model': return <ModelIntelligence data={metricsData} />;
      case 'rules': return renderRules();
      default: return renderCommandCenter();
    }
  };

  return (
    <div className="app-shell">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

      <div className="main-content">
        <TopBar activeTab={activeTab} />

        <div className="workspace">{renderActiveTab()}</div>
      </div>

      <div className={`drawer ${isDrawerOpen ? 'open' : ''}`}>
        <button className="drawer-close" onClick={() => setIsDrawerOpen(false)} aria-label="Close drawer">
          <X size={20} />
        </button>
        {predictionData && (
          <IncidentDrawer
            data={predictionData}
            simulation={simulationData}
            optimization={optimizationData}
          />
        )}
      </div>
    </div>
  );
}

export default App;
