import React, { useState, useRef, useCallback } from 'react';
import Navbar from './components/Navbar';
import VideoPanel from './components/VideoPanel';
import SidePanel from './components/SidePanel';

// ── Tuned thresholds (lowered from original overly-strict values) ────────────
const HISTORY_MIN_CONFIDENCE = 30;     // was 85 — too strict with network latency
const BUFFER_MIN_CONFIDENCE = 27;      // was 93 — way too strict
const STABLE_WINDOW_MS = 500;          // was 700 — more responsive
const CHAR_COOLDOWN_MS = 400;          // Cooldown between different characters
const REPEAT_COOLDOWN_MS = 2000;       // Repeat delay for same character

function App() {
  const [activeModel, setActiveModel] = useState('RF');
  const [history, setHistory] = useState([]);
  const [translatedText, setTranslatedText] = useState("");

  // ── FIX (Bug #4): Use refs instead of state for candidate tracking ────────
  // State values captured inside memoized callbacks become stale.
  // Refs always point to the latest value.
  const lastCandidateRef = useRef({ char: null, firstSeenAt: 0 });
  const lastCommittedAtRef = useRef(0);
  const lastCommittedCharRef = useRef(null);

  const handlePrediction = useCallback((char, confidence) => {
    if (!char) {
      lastCandidateRef.current = { char: null, firstSeenAt: 0 };
      lastCommittedCharRef.current = null;
      return;
    }

    const now = Date.now();
    const candidate = lastCandidateRef.current;

    // New character detected — start the stability window
    if (char !== candidate.char) {
      lastCandidateRef.current = { char, firstSeenAt: now };
      return;
    }

    // Same character but hasn't been stable long enough yet
    if (now - candidate.firstSeenAt < STABLE_WINDOW_MS) return;

    // Character has been stable for STABLE_WINDOW_MS — process it

    // Add to history if above history threshold
    if (confidence >= HISTORY_MIN_CONFIDENCE) {
      setHistory((prev) => {
        if (prev[0]?.char !== char) {
          return [{ char, confidence, ts: now }, ...prev].slice(0, 15);
        }
        return prev;
      });
    }

    // Commit to translated text if above buffer threshold and cooldown elapsed
    const cooldown = (char === lastCommittedCharRef.current) ? REPEAT_COOLDOWN_MS : CHAR_COOLDOWN_MS;

    if (confidence >= BUFFER_MIN_CONFIDENCE && (now - lastCommittedAtRef.current) >= cooldown) {
      setTranslatedText((prev) => prev + char);
      lastCommittedAtRef.current = now;
      lastCommittedCharRef.current = char;
      // NOTE: We don't reset lastCandidateRef here anymore. 
      // This keeps the stability window open while the character is held.
    }
  }, []);

  const actions = {
    addSpace: () => setTranslatedText(prev => prev + " "),
    deleteLast: () => setTranslatedText(prev => prev.slice(0, -1)),
    clearAll: () => {
      setTranslatedText("");
      setHistory([]);
      lastCandidateRef.current = { char: null, firstSeenAt: 0 };
      lastCommittedAtRef.current = 0;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans selection:bg-brand-cyan selection:text-white text-slate-900">
      <Navbar />

      <main className="flex-1 max-w-[1600px] w-full mx-auto p-6 lg:p-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 h-full">

          {/* Main Video Section */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            <VideoPanel
              activeModel={activeModel}
              onPrediction={handlePrediction}
            />
          </div>

          {/* Sidebar Section */}
          <div className="lg:col-span-4 flex flex-col h-full">
            <SidePanel
              activeModel={activeModel}
              setActiveModel={setActiveModel}
              history={history}
              translatedText={translatedText}
              actions={actions}
            />
          </div>

        </div>
      </main>

      {/* Footer Info */}
      <footer className="px-10 py-4 text-center text-slate-400 text-xs font-medium uppercase tracking-tight">
        Powered by SignVerse AI • ISL Translation System v1.0
      </footer>
    </div>
  );
}

export default App;
