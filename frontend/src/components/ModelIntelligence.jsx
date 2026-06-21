import React from 'react';
import { Cpu, TrendingUp, Target, AlertTriangle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, LineChart, Line, Cell,
} from 'recharts';

const COLORS = {
  critical: '#FF5D5D',
  moderate: '#FFB84D',
  minor: '#47D18C',
  focus: '#9EC1FF',
  muted: '#3A4052',
  text: '#A7B0BE',
};

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#1a1a2e', border: '1px solid #3A4052',
      borderRadius: 6, padding: '8px 12px', fontSize: 12,
    }}>
      <div style={{ color: '#F5F7FA', marginBottom: 4, fontWeight: 600 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || COLORS.focus }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  );
};

const ModelIntelligence = ({ data }) => {
  if (!data) {
    return (
      <div className="panel" style={{ padding: 'var(--space-5)', textAlign: 'center' }}>
        <Cpu size={48} className="text-muted" />
        <h2 className="text-md" style={{ marginTop: 'var(--space-3)' }}>Connecting to API...</h2>
        <p className="text-muted text-sm">Start the backend to view model intelligence</p>
      </div>
    );
  }

  const model = data.model || {};
  const learning = data.learning || {};

  // Confusion matrix-style data
  const classMetrics = [
    { class: '<30min', precision: model['<30min']?.precision || 0, recall: model['<30min']?.recall || 0, f1: model['<30min']?.['f1-score'] || 0, support: model['<30min']?.support || 0 },
    { class: '30min-2hr', precision: model['30min-2hr']?.precision || 0, recall: model['30min-2hr']?.recall || 0, f1: model['30min-2hr']?.['f1-score'] || 0, support: model['30min-2hr']?.support || 0 },
    { class: '2hr+', precision: model['2hr+']?.precision || 0, recall: model['2hr+']?.recall || 0, f1: model['2hr+']?.['f1-score'] || 0, support: model['2hr+']?.support || 0 },
  ];

  const metricsChartData = classMetrics.map(cm => ({
    name: cm.class,
    Precision: +(cm.precision * 100).toFixed(1),
    Recall: +(cm.recall * 100).toFixed(1),
    F1: +(cm.f1 * 100).toFixed(1),
  }));

  // Post-event learning trend
  const accuracyTrend = learning.accuracy_trend || [];
  const corridorBias = learning.corridor_stats || {};
  const biasedCorridors = Object.entries(corridorBias)
    .filter(([, s]) => s.bias_direction !== 'calibrated' && s.count >= 3)
    .sort((a, b) => Math.abs(b[1].mean_error_min) - Math.abs(a[1].mean_error_min))
    .slice(0, 8)
    .map(([name, s]) => ({
      corridor: name.length > 14 ? name.substring(0, 12) + '…' : name,
      error: s.mean_error_min,
      abs_error: s.mean_abs_error_min,
      count: s.count,
      direction: s.bias_direction,
    }));

  // Predicted vs actual scatter (from learning records)
  const scatterData = (learning.records || []).slice(0, 50).map(r => ({
    actual: r.actual_duration_min,
    predicted: r.predicted_duration_min,
    correct: r.correct,
  }));

  return (
    <div className="analytics-grid">
      {/* Model Overview */}
      <div className="panel chart-card" style={{ gridColumn: 'span 2' }}>
        <div className="section-heading">
          <Cpu size={18} />
          <h2>Model Performance</h2>
        </div>
        <div className="stat-strip compact" style={{ marginTop: 'var(--space-3)' }}>
          <div className="stat-tile mini">
            <span>Accuracy</span>
            <strong>{model.accuracy ? (model.accuracy * 100).toFixed(1) + '%' : '--'}</strong>
          </div>
          <div className="stat-tile mini">
            <span>Macro F1</span>
            <strong>{model.macro_f1?.toFixed(3) || '--'}</strong>
          </div>
          <div className="stat-tile mini">
            <span>Critical Recall</span>
            <strong className="severity-critical">{model.critical_recall ? (model.critical_recall * 100).toFixed(1) + '%' : '--'}</strong>
          </div>
          <div className="stat-tile mini">
            <span>Ensemble</span>
            <strong>{(model.selected_models || []).length} models</strong>
          </div>
        </div>
        {model.selected_models && (
          <div className="text-xs text-muted" style={{ marginTop: 'var(--space-2)' }}>
            Models: {model.selected_models.join(', ')}
          </div>
        )}
        {model.kfold_validation && (
          <div className="text-xs text-muted" style={{ marginTop: 'var(--space-1)' }}>
            3-Fold CV: {(model.kfold_validation.accuracy_mean * 100).toFixed(1)}% ± {(model.kfold_validation.accuracy_std * 100).toFixed(1)}% accuracy,
            {(model.kfold_validation.critical_recall_mean * 100).toFixed(1)}% critical recall
          </div>
        )}
      </div>

      {/* Per-Class Metrics Chart */}
      <div className="panel chart-card">
        <h3 className="label">Per-Class Precision / Recall / F1</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={metricsChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis dataKey="name" tick={{ fill: COLORS.text, fontSize: 11 }} />
            <YAxis tick={{ fill: COLORS.text, fontSize: 11 }} domain={[0, 100]} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="Precision" fill={COLORS.focus} radius={[3, 3, 0, 0]} barSize={16} />
            <Bar dataKey="Recall" fill={COLORS.moderate} radius={[3, 3, 0, 0]} barSize={16} />
            <Bar dataKey="F1" fill={COLORS.minor} radius={[3, 3, 0, 0]} barSize={16} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Predicted vs Actual Scatter */}
      <div className="panel chart-card">
        <h3 className="label">
          <Target size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
          Predicted vs Actual Duration
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis dataKey="actual" name="Actual (min)" tick={{ fill: COLORS.text, fontSize: 10 }} />
            <YAxis dataKey="predicted" name="Predicted (min)" tick={{ fill: COLORS.text, fontSize: 10 }} />
            <Tooltip content={<ChartTooltip />} />
            <Scatter data={scatterData} fill={COLORS.focus} fillOpacity={0.6} r={3} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Post-Event Learning Trend */}
      <div className="panel chart-card">
        <h3 className="label">
          <TrendingUp size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
          Learning Curve (Accuracy Over Time)
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={accuracyTrend}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis dataKey="window_start" tick={{ fill: COLORS.text, fontSize: 10 }} label={{ value: 'Events', fill: COLORS.text, fontSize: 10, position: 'bottom' }} />
            <YAxis tick={{ fill: COLORS.text, fontSize: 10 }} domain={[0, 1]} />
            <Tooltip content={<ChartTooltip />} />
            <Line type="monotone" dataKey="accuracy" stroke={COLORS.minor} strokeWidth={2} dot={{ fill: COLORS.minor, r: 3 }} name="Accuracy" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Corridor Bias */}
      <div className="panel chart-card">
        <h3 className="label">
          <AlertTriangle size={14} style={{ marginRight: 4, verticalAlign: 'middle' }} />
          Per-Corridor Prediction Bias (min)
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={biasedCorridors} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.muted} />
            <XAxis type="number" tick={{ fill: COLORS.text, fontSize: 10 }} />
            <YAxis type="category" dataKey="corridor" tick={{ fill: COLORS.text, fontSize: 10 }} width={110} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="error" name="Bias (min)" radius={[0, 3, 3, 0]}>
              {biasedCorridors.map((entry, i) => (
                <Cell key={i} fill={entry.error > 0 ? COLORS.moderate : COLORS.focus} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Learning Summary */}
      <div className="panel chart-card">
        <h3 className="label">Post-Event Learning Summary</h3>
        <div className="stat-strip compact" style={{ marginTop: 'var(--space-2)' }}>
          <div className="stat-tile mini">
            <span>Events Tracked</span>
            <strong>{learning.total_events || '--'}</strong>
          </div>
          <div className="stat-tile mini">
            <span>Accuracy</span>
            <strong>{learning.overall_accuracy ? (learning.overall_accuracy * 100).toFixed(1) + '%' : '--'}</strong>
          </div>
          <div className="stat-tile mini">
            <span>MAE</span>
            <strong>{learning.mean_absolute_error_min ? learning.mean_absolute_error_min + ' min' : '--'}</strong>
          </div>
        </div>
        <div className="rec-card" style={{ marginTop: 'var(--space-3)' }}>
          <div className="text-xs text-muted" style={{ marginBottom: 4 }}>System Bias</div>
          <div className="text-sm font-semibold">{learning.bias_direction || 'Not computed'}</div>
          <div className="text-xs text-muted" style={{ marginTop: 4 }}>
            Mean error: {learning.mean_error_min || '--'} min •
            Median abs error: {learning.median_abs_error_min || '--'} min
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelIntelligence;
