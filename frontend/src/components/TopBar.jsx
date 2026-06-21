import React from 'react';
import { Clock } from 'lucide-react';

const TAB_TITLES = {
  command: 'Command Center',
  simulate: 'Impact Simulator',
  optimizer: 'Resource Optimizer',
  analytics: 'Analytics Dashboard',
  network: 'Network Graph',
  corridors: 'Corridor Intelligence',
  model: 'Model Intelligence',
  rules: 'Operational Protocol',
};

const TopBar = ({ activeTab }) => {
  return (
    <div className="topbar">
      <h1 className="topbar-title">{TAB_TITLES[activeTab] || 'Flow 2.0'}</h1>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1 text-xs text-muted">
          <Clock size={14} />
          <span>Live • {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <span className="topbar-badge">v2.0</span>
      </div>
    </div>
  );
};

export default TopBar;
