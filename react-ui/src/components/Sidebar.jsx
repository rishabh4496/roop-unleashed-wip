import React from 'react';
import { Settings, BarChart2, List, CreditCard, FileText, Award, ArrowUpCircle } from 'lucide-react';

const Sidebar = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart2 },
    { id: 'transactions', label: 'Transactions', icon: List },
    { id: 'cards', label: 'Cards', icon: CreditCard },
    { id: 'invoices', label: 'Invoices', icon: FileText },
    { id: 'goals', label: 'Goals', icon: Award },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="sidebar">
      <div className="user-profile">
        <img 
          src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-1.2.1&auto=format&fit=crop&w=256&q=80" 
          alt="User Avatar" 
          className="user-avatar"
        />
        <div className="user-name">Roop User</div>
        <div className="user-role">Premium Account</div>
      </div>
      
      <div className="nav-links">
        {tabs.map(tab => {
          const Icon = tab.icon;
          return (
            <div 
              key={tab.id}
              className={`nav-link ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon size={20} />
              {tab.label}
            </div>
          );
        })}
      </div>
      
      <div style={{ marginTop: 'auto' }}>
        <div className="nav-link" style={{ background: 'rgba(255,255,255,0.1)', justifyContent: 'center' }}>
          <span>Upgrade to premium</span>
          <ArrowUpCircle size={16} style={{ marginLeft: 4 }} />
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
