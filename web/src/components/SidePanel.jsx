import React from 'react';

const SidePanel = ({ activeModel, setActiveModel, history, translatedText, actions }) => {
  const models = [
    { id: 'MLP', label: 'MLP' },
    { id: 'RF', label: 'Random Forest' },
    { id: 'KNN', label: 'KNN' }
  ];

  return (
    <div className="flex flex-col gap-6 h-full">
      {/* Model Selection */}
      <div className="p-6 bg-white rounded-3xl border border-slate-100 shadow-xl">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">AI Model Engine</h3>
        <div className="grid grid-cols-1 gap-2">
          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => setActiveModel(model.id)}
              className={`flex items-center justify-between px-4 py-3 rounded-2xl border transition-all duration-300 font-bold ${
                activeModel === model.id
                  ? 'bg-brand-cyan/10 border-brand-cyan text-brand-cyan shadow-sm'
                  : 'bg-slate-50 border-transparent text-slate-500 hover:bg-slate-100'
              }`}
            >
              <span>{model.label}</span>
              {activeModel === model.id && (
                <div className="w-2 h-2 rounded-full bg-brand-cyan animate-pulse"></div>
              )}
            </button>
          ))}
        </div>
      </div>
      
      {/* Recent History */}
      <div className="p-6 bg-white rounded-3xl border border-slate-100 shadow-lg">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Recent Detects</h3>
        <div className="flex gap-2 flex-wrap">
          {history.length > 0 ? (
            history.map((item, index) => (
              <div key={index} className="relative w-10 h-10 flex items-center justify-center rounded-xl border-2 border-brand-cyan/30 text-brand-cyan font-black text-lg bg-brand-cyan/5">
                {item.char}
                <span className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[9px] text-slate-400 font-mono">
                  {Math.round(item.confidence)}%
                </span>
              </div>
            ))
          ) : (
             <div className="text-xs font-medium text-slate-300 italic py-2">No history yet</div>
          )}
        </div>
      </div>
      
      {/* Translated Text Area */}
      <div className="flex-1 p-6 bg-slate-900 rounded-3xl shadow-2xl flex flex-col gap-4 border border-slate-800">
        <div className="flex justify-between items-center px-1">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest">Translation Buffer</h3>
          <span className="text-[10px] text-slate-400 font-mono uppercase">{translatedText.length} chars</span>
        </div>
        
        <div className="flex-1 bg-transparent text-white font-bold text-3xl p-2 outline-none resize-none leading-relaxed overflow-y-auto break-all scrollbar-hide">
          {translatedText || "..."}
        </div>
        
        {/* Actions Buttons */}
        <div className="grid grid-cols-3 gap-3">
          <button 
            onClick={actions.addSpace}
            className="py-3 px-2 bg-slate-800 hover:bg-slate-700 text-white font-bold rounded-2xl border border-slate-700 transition-all active:scale-95 shadow-lg text-xs"
          >
            Space
          </button>
          <button 
            onClick={actions.deleteLast}
            className="py-3 px-2 bg-yellow-600/10 hover:bg-yellow-600/20 text-yellow-500 font-bold rounded-2xl border border-yellow-600/30 transition-all active:scale-95 shadow-lg text-xs"
          >
            Delete
          </button>
          <button 
            onClick={actions.clearAll}
            className="py-3 px-2 bg-red-600/10 hover:bg-red-600/20 text-red-500 font-bold rounded-2xl border border-red-600/30 transition-all active:scale-95 shadow-lg text-xs"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
};

export default SidePanel;

