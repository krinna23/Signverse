import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = 'http://localhost:5000';

const Navbar = () => {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [loadedModels, setLoadedModels] = useState([]);

  useEffect(() => {
    let interval;
    const checkHealth = async () => {
      try {
        const res = await axios.get(`${BACKEND_URL}/health`, { timeout: 2000 });
        setBackendStatus('connected');
        setLoadedModels(res.data.models_loaded || []);
      } catch {
        setBackendStatus('offline');
        setLoadedModels([]);
      }
    };

    checkHealth();
    interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const statusConfig = {
    connected: {
      bg: 'bg-green-50',
      text: 'text-green-700',
      border: 'border-green-100',
      dot: 'bg-green-500',
      ping: 'bg-green-400',
      label: 'Connected',
    },
    offline: {
      bg: 'bg-red-50',
      text: 'text-red-700',
      border: 'border-red-100',
      dot: 'bg-red-500',
      ping: 'bg-red-400',
      label: 'Offline',
    },
    checking: {
      bg: 'bg-yellow-50',
      text: 'text-yellow-700',
      border: 'border-yellow-100',
      dot: 'bg-yellow-500',
      ping: 'bg-yellow-400',
      label: 'Checking...',
    },
  };

  const s = statusConfig[backendStatus];

  return (
    <nav className="flex items-center justify-between px-8 py-4 bg-white border-b border-gray-100 shadow-sm">
      <div className="flex items-center gap-2">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">
          Sign<span className="text-brand-cyan">Verse</span>
        </h1>
      </div>
      
      <div className="hidden md:flex items-center gap-3 text-slate-500 font-medium text-sm lg:text-base">
        <span>Real-time Indian Sign Language Translation</span>
        {loadedModels.length > 0 && (
          <span className="text-xs text-slate-400 font-mono">
            ({loadedModels.join(', ')})
          </span>
        )}
      </div>
      
      <div className="flex items-center gap-3">
        <div className={`flex items-center gap-2 px-3 py-1.5 ${s.bg} ${s.text} rounded-full border ${s.border} text-sm font-semibold`}>
          <span className="relative flex h-2 w-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${s.ping} opacity-75`}></span>
            <span className={`relative inline-flex rounded-full h-2 w-2 ${s.dot}`}></span>
          </span>
          {s.label}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
