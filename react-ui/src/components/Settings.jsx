import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Save, RefreshCw, Trash2, CheckCircle, AlertTriangle } from 'lucide-react';

const Settings = () => {
  const [settings, setSettings] = useState({
    server_share: false,
    clear_output: false,
    output_template: '{file}_{timestamp}',
    server_name: '',
    server_port: 0,
    provider: 'cpu',
    default_det_size: true,
    force_cpu: false,
    max_threads: 3,
    memory_limit: 0,
    output_image_format: 'png',
    output_video_codec: 'libx264',
    output_video_format: 'mp4',
    video_quality: 14,
    use_os_temp_folder: false,
    output_show_video: false
  });

  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : type === 'number' ? Number(value) : value
    }));
  };

  const resetToDefaults = () => {
    setSettings({
      server_share: false,
      clear_output: false,
      output_template: '{file}_{timestamp}',
      server_name: '',
      server_port: 0,
      provider: 'cpu',
      default_det_size: true,
      force_cpu: false,
      max_threads: 3,
      memory_limit: 0,
      output_image_format: 'png',
      output_video_codec: 'libx264',
      output_video_format: 'mp4',
      video_quality: 14,
      use_os_temp_folder: false,
      output_show_video: false
    });
    showToast('Settings reset to defaults', 'info');
  };

  const applySettings = async () => {
    try {
      // In a real app, you would fetch to your FastAPI endpoint here.
      // await fetch('/api/settings', { method: 'POST', body: JSON.stringify(settings) })
      showToast('Settings applied successfully!');
    } catch (error) {
      showToast('Failed to apply settings', 'error');
    }
  };

  const cleanTemp = () => {
    showToast('Temp folder cleaned', 'success');
  };

  const restartServer = () => {
    showToast('Server restarting...', 'info');
  };

  return (
    <div className="settings-container">
      <div className="header">
        <h1>Settings</h1>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn btn-secondary" onClick={cleanTemp}>
            <Trash2 size={16} /> Clean Temp
          </button>
          <button className="btn btn-danger" onClick={restartServer}>
            <RefreshCw size={16} /> Restart Server
          </button>
        </div>
      </div>

      <div className="settings-grid">
        {/* Core Settings */}
        <div className="glass-panel">
          <div className="card-title">
            <SettingsIcon size={20} />
            General
          </div>
          
          <div className="form-group">
            <label className="toggle-switch">
              <input type="checkbox" name="server_share" checked={settings.server_share} onChange={handleChange} />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Public Server</span>
            </label>
          </div>
          
          <div className="form-group">
            <label className="toggle-switch">
              <input type="checkbox" name="clear_output" checked={settings.clear_output} onChange={handleChange} />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Clear output folder before each run</span>
            </label>
          </div>

          <div className="form-group">
            <label>Filename Output Template</label>
            <span className="info">Tokens: {'{file}'}, {'{time}'}, {'{index}'}, {'{timestamp}'}</span>
            <input type="text" name="output_template" value={settings.output_template} onChange={handleChange} />
          </div>

          <div className="form-group">
            <label>Server Name</label>
            <span className="info">Leave blank to run locally</span>
            <input type="text" name="server_name" value={settings.server_name} onChange={handleChange} />
          </div>

          <div className="form-group">
            <label>Server Port</label>
            <span className="info">Leave at 0 to use default</span>
            <input type="number" name="server_port" value={settings.server_port} onChange={handleChange} />
          </div>
        </div>

        {/* Processing Settings */}
        <div className="glass-panel">
          <div className="card-title">
            <SettingsIcon size={20} />
            Processing
          </div>

          <div className="form-group">
            <label>Execution Provider</label>
            <select name="provider" value={settings.provider} onChange={handleChange}>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
              <option value="tensorrt">TensorRT</option>
            </select>
          </div>

          <div className="form-group">
            <label className="toggle-switch">
              <input type="checkbox" name="default_det_size" checked={settings.default_det_size} onChange={handleChange} />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Use default Det-Size</span>
            </label>
          </div>

          <div className="form-group">
            <label className="toggle-switch">
              <input type="checkbox" name="force_cpu" checked={settings.force_cpu} onChange={handleChange} />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Force CPU for Face Analyser</span>
            </label>
          </div>

          <div className="form-group">
            <label>Max. Number of Threads: {settings.max_threads}</label>
            <input type="range" name="max_threads" min="1" max="32" value={settings.max_threads} onChange={handleChange} />
          </div>

          <div className="form-group">
            <label>Max. Memory to use (Gb): {settings.memory_limit === 0 ? 'No Limit' : settings.memory_limit}</label>
            <input type="range" name="memory_limit" min="0" max="128" value={settings.memory_limit} onChange={handleChange} />
          </div>
        </div>

        {/* Media Settings */}
        <div className="glass-panel">
          <div className="card-title">
            <SettingsIcon size={20} />
            Media Output
          </div>

          <div className="form-group">
            <label>Image Output Format</label>
            <select name="output_image_format" value={settings.output_image_format} onChange={handleChange}>
              <option value="png">PNG</option>
              <option value="jpg">JPG</option>
              <option value="webp">WEBP</option>
            </select>
          </div>

          <div className="form-group">
            <label>Video Codec</label>
            <select name="output_video_codec" value={settings.output_video_codec} onChange={handleChange}>
              <option value="libx264">libx264</option>
              <option value="libx265">libx265</option>
              <option value="libvpx-vp9">libvpx-vp9</option>
              <option value="h264_nvenc">h264_nvenc</option>
              <option value="hevc_nvenc">hevc_nvenc</option>
            </select>
          </div>

          <div className="form-group">
            <label>Video Output Format</label>
            <select name="output_video_format" value={settings.output_video_format} onChange={handleChange}>
              <option value="mp4">MP4</option>
              <option value="avi">AVI</option>
              <option value="mkv">MKV</option>
              <option value="webm">WEBM</option>
            </select>
          </div>

          <div className="form-group">
            <label>Video Quality (crf): {settings.video_quality}</label>
            <input type="range" name="video_quality" min="0" max="100" value={settings.video_quality} onChange={handleChange} />
          </div>

          <div className="form-group">
            <label className="toggle-switch">
              <input type="checkbox" name="use_os_temp_folder" checked={settings.use_os_temp_folder} onChange={handleChange} />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Use OS temp folder</span>
            </label>
          </div>

          <div className="form-group">
            <label className="toggle-switch">
              <input type="checkbox" name="output_show_video" checked={settings.output_show_video} onChange={handleChange} />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Show video in browser (re-encodes)</span>
            </label>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '16px', marginTop: '32px' }}>
        <button className="btn btn-secondary" onClick={resetToDefaults}>
          Reset to Defaults
        </button>
        <button className="btn btn-primary" onClick={applySettings}>
          <Save size={18} /> Apply Settings
        </button>
      </div>

      {toast && (
        <div className="toast-container">
          <div className={`toast ${toast.type}`}>
            {toast.type === 'success' ? <CheckCircle size={20} /> : <AlertTriangle size={20} />}
            {toast.message}
          </div>
        </div>
      )}
    </div>
  );
};

export default Settings;
