import React from 'react';
import { LayoutDashboard, Zap, Users, BarChart3, Share2, MapPin, Cpu, Settings, Navigation } from 'lucide-react';

const Sidebar = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'command', label: 'Command Center', icon: LayoutDashboard },
    { id: 'simulate', label: 'Impact Simulator', icon: Zap },
    { id: 'optimizer', label: 'Resource Optimizer', icon: Users },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'network', label: 'Network Graph', icon: Share2 },
    { id: 'corridors', label: 'Corridors', icon: MapPin },
    { id: 'model', label: 'Model Intelligence', icon: Cpu },
    { id: 'rules', label: 'Protocol', icon: Settings },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="logo-container">
          <Navigation size={24} className="logo-icon" />
          <span className="logo-text">FLOW</span>
        </div>
      </div>
      <div className="nav-list">
        {navItems.map(item => {
          const Icon = item.icon;
          return (
            <div
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') setActiveTab(item.id);
              }}
            >
              <Icon size={18} />
              <span className="text-sm font-medium">{item.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Sidebar;
