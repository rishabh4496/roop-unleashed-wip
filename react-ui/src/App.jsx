import React from 'react';
import FaceSwap from './components/FaceSwap';

function App() {
  return (
    <div className="min-h-screen">
      <header className="p-6 border-b border-white/10 bg-black/20 backdrop-blur-md">
        <h1 className="text-3xl font-bold text-white tracking-wide">Roop Unleashed <span className="text-[#E94560]">Pro</span></h1>
      </header>
      <main className="max-w-7xl mx-auto py-8">
        <FaceSwap />
      </main>
    </div>
  );
}

export default App;
