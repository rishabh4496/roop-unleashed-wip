import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Settings from './components/Settings';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('settings');
  const [theme, setTheme] = useState('purple');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const renderContent = () => {
    switch (activeTab) {
      case 'settings':
        return <Settings />;
      default:
        return (
          <div className="main-content" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <h2>{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Dashboard</h2>
          </div>
        );
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <div className="main-content">
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '20px' }}>
          <div className="theme-switcher">
            <button 
              className={`theme-btn ${theme === 'light' ? 'active' : ''}`}
              onClick={() => setTheme('light')}
            >
              Light
            </button>
            <button 
              className={`theme-btn ${theme === 'purple' ? 'active' : ''}`}
              onClick={() => setTheme('purple')}
            >
              Purple
            </button>
            <button 
              className={`theme-btn ${theme === 'dark' ? 'active' : ''}`}
              onClick={() => setTheme('dark')}
            >
              Dark
            </button>
          </div>
        </div>

        {renderContent()}
      </div>
    </div>
  );
}

export default App;
