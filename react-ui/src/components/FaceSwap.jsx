import React, { useState } from 'react';

const FaceSwap = () => {
  const [sourceFile, setSourceFile] = useState(null);
  const [targetFile, setTargetFile] = useState(null);
  const [sourcePath, setSourcePath] = useState('');
  const [targetPath, setTargetPath] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');

  const handleSourceUpload = async (e) => {
    const file = e.target.files[0];
    setSourceFile(file);
    const formData = new FormData();
    formData.append('file', file);
    
    setStatus('Uploading source...');
    const res = await fetch('http://127.0.0.1:8001/api/upload_source', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    setSourcePath(data.path);
    setStatus('Source uploaded.');
  };

  const handleTargetUpload = async (e) => {
    const file = e.target.files[0];
    setTargetFile(file);
    const formData = new FormData();
    formData.append('file', file);
    
    setStatus('Uploading target...');
    const res = await fetch('http://127.0.0.1:8001/api/upload_target', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    setTargetPath(data.path);
    setStatus('Target uploaded.');
  };

  const startSwap = async () => {
    if (!sourcePath || !targetPath) {
      alert("Please upload both source and target files.");
      return;
    }
    
    setIsProcessing(true);
    setStatus('Starting process...');
    
    const res = await fetch('http://127.0.0.1:8001/api/swap', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_path: sourcePath, target_path: targetPath })
    });
    
    if (res.ok) {
      setStatus('Processing started in background! Check terminal for logs.');
    } else {
      setStatus('Failed to start processing.');
    }
    setIsProcessing(false);
  };

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold mb-6">Face Swap</h2>
      
      <div className="grid grid-cols-2 gap-8 mb-8">
        <div className="p-6 rounded-xl bg-white/10 backdrop-blur-md border border-white/20">
          <h3 className="text-xl mb-4">Source Image</h3>
          <input type="file" onChange={handleSourceUpload} accept="image/*" className="mb-4 block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-white/10 file:text-white hover:file:bg-white/20"/>
          {sourceFile && <p className="text-green-400 text-sm">Loaded: {sourceFile.name}</p>}
        </div>

        <div className="p-6 rounded-xl bg-white/10 backdrop-blur-md border border-white/20">
          <h3 className="text-xl mb-4">Target Video/Image</h3>
          <input type="file" onChange={handleTargetUpload} accept="video/*,image/*" className="mb-4 block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-white/10 file:text-white hover:file:bg-white/20"/>
          {targetFile && <p className="text-green-400 text-sm">Loaded: {targetFile.name}</p>}
        </div>
      </div>

      <button 
        onClick={startSwap} 
        disabled={isProcessing}
        className="w-full py-4 rounded-xl font-bold text-lg bg-[#E94560] hover:bg-[#D63447] transition-all shadow-[0_4px_15px_rgba(233,69,96,0.4)] disabled:opacity-50"
      >
        {isProcessing ? 'Processing...' : 'Start Swapping'}
      </button>

      {status && (
        <div className="mt-6 p-4 rounded-lg bg-black/30 border border-white/10">
          <p className="text-center">{status}</p>
        </div>
      )}
    </div>
  );
};

export default FaceSwap;
