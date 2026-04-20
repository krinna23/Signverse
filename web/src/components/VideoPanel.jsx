import React, { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import axios from 'axios';

// ── Hand skeleton connections (matches camera_test.py HAND_CONNECTIONS) ───────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// ── Pin the WASM version to match our npm package ────────────────────────────
const MEDIAPIPE_WASM_VERSION = "0.10.34";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_WASM_VERSION}/wasm`;

const BACKEND_URL = 'http://localhost:5000';
const INFERENCE_INTERVAL_MS = 150;
const FEATURES_PER_HAND = 63;

const VideoPanel = ({ activeModel = 'RF', onPrediction }) => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const lastInferenceAtRef = useRef(0);
  const inFlightRef = useRef(false);
  const activeModelRef = useRef(activeModel);

  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [statusMessage, setStatusMessage] = useState("No Hand Detected");
  const [isLoading, setIsLoading] = useState(true);
  const [handLandmarker, setHandLandmarker] = useState(null);
  const [handsDetected, setHandsDetected] = useState(0);

  // Keep ref in sync with prop
  useEffect(() => {
    activeModelRef.current = activeModel;
  }, [activeModel]);

  // ── Initialize MediaPipe ──────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        if (!cancelled) {
          setHandLandmarker(landmarker);
          setIsLoading(false);
        }
      } catch (err) {
        console.error("Failed to initialize MediaPipe:", err);
        if (!cancelled) {
          setStatusMessage("MediaPipe Init Failed");
          setIsLoading(false);
        }
      }
    };
    initMediaPipe();
    return () => { cancelled = true; };
  }, []);

  // ── Draw hand skeleton ────────────────────────────────────────────────────
  // Canvas does NOT have CSS scale-x-[-1] anymore.
  // We manually mirror x-coordinates so text renders correctly (not backwards).
  const drawResults = useCallback((results) => {
    if (!canvasRef.current || !webcamRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    const video = webcamRef.current.video;
    const W = video.videoWidth;
    const H = video.videoHeight;

    canvasRef.current.width = W;
    canvasRef.current.height = H;
    ctx.clearRect(0, 0, W, H);

    if (!results.landmarks || results.landmarks.length === 0) return;

    for (let i = 0; i < results.landmarks.length; i++) {
      const landmarks = results.landmarks[i];
      const rawHandedness = results.handedness[i]?.[0]?.categoryName || "Unknown";

      // Convert to pixel coords, MIRROR x to match the mirrored webcam display
      const pts = landmarks.map(lm => ({
        x: W - (lm.x * W),   // mirror x so overlay aligns with mirrored webcam
        y: lm.y * H,
      }));

      // Draw connection lines
      ctx.strokeStyle = "rgba(180, 0, 255, 0.8)";
      ctx.lineWidth = 2;
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
        ctx.stroke();
      }

      // Draw landmark points
      ctx.fillStyle = "#00FFB4";
      for (const pt of pts) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Draw hand label above wrist — show the RAW MediaPipe label
      // MediaPipe's handedness is anatomy-based, so "Left" = user's left hand
      const wrist = pts[0];
      ctx.font = "bold 14px Inter, sans-serif";
      ctx.fillStyle = "#FFE040";
      ctx.textAlign = "center";
      ctx.fillText(rawHandedness, wrist.x, wrist.y - 15);
    }
  }, []);

  // ── Extract features & send to backend ────────────────────────────────────
  // CRITICAL: camera_test.py does cv2.flip(frame, 1) BEFORE MediaPipe detection.
  // This means:
  //   1. Landmark x-coordinates are MIRRORED (x → 1-x in normalized space)
  //   2. Handedness labels are SWAPPED (left hand looks like right after flip)
  //
  // The training data (preprocess.py) uses raw images (no flip), BUT the model
  // works with camera_test.py which flips. This means camera_test.py's flip
  // produces features that work with the trained model.
  //
  // To match camera_test.py from the web app (which doesn't flip the actual frame),
  // we must simulate the flip mathematically:
  //   - Flip x coords: use (1.0 - lm.x) instead of lm.x
  //   - Swap handedness: "Left" → "Right", "Right" → "Left"
  const processDetection = useCallback(async (results) => {
    if (!results.landmarks || results.landmarks.length === 0) {
      setPrediction("");
      setConfidence(0);
      setHandsDetected(0);
      setStatusMessage("No Hand Detected");
      if (onPrediction) onPrediction(null, 0);
      return;
    }

    setHandsDetected(results.landmarks.length);

    const formattedLandmarks = {};
    for (let idx = 0; idx < results.landmarks.length; idx++) {
      const lms = results.landmarks[idx];
      const rawSide = results.handedness[idx][0].categoryName;

      // SWAP handedness to match camera_test.py's flipped detection
      const side = rawSide === "Left" ? "Right" : "Left";

      // FLIP x-coordinates to simulate cv2.flip(frame, 1)
      // After flip: x_flipped = 1.0 - x
      // Normalize: x_flipped - min(x_flipped) = (1-x) - (1-max_x) = max_x - x
      const xs = lms.map(l => l.x);
      const ys = lms.map(l => l.y);
      const maxX = Math.max(...xs);   // needed for flipped normalization
      const minY = Math.min(...ys);

      const features = [];
      for (const l of lms) {
        features.push(maxX - l.x);    // flipped & normalized x (equivalent to camera_test.py)
        features.push(l.y - minY);    // normalized y (same as camera_test.py)
        features.push(l.z);           // z unchanged
      }

      if (features.length === FEATURES_PER_HAND) {
        formattedLandmarks[side] = features;
      }
    }

    if (Object.keys(formattedLandmarks).length === 0) {
      setPrediction("");
      setConfidence(0);
      setStatusMessage("Landmarks Invalid");
      return;
    }

    try {
      inFlightRef.current = true;
      const response = await axios.post(`${BACKEND_URL}/predict`, {
        landmarks: formattedLandmarks,
        model_type: activeModelRef.current,
      }, { timeout: 2000 });

      if (response.data.prediction) {
        const { prediction: char, confidence: conf } = response.data;
        setPrediction(char);
        setConfidence(conf);
        setStatusMessage("");
        if (onPrediction) {
          onPrediction(char, conf);
        }
      } else {
        setPrediction("");
        setConfidence(0);
        setStatusMessage("No Prediction From Model");
      }
    } catch (err) {
      console.error("Prediction error:", err);
      setPrediction("");
      setConfidence(0);
      if (err.code === 'ECONNABORTED') {
        setStatusMessage("Backend Timeout");
      } else if (err.response) {
        setStatusMessage(`Backend Error (${err.response.status})`);
      } else {
        setStatusMessage("Backend Offline — start app.py");
      }
      if (onPrediction) onPrediction(null, 0);
    } finally {
      inFlightRef.current = false;
    }
  }, [onPrediction]);

  // ── Main detection loop ───────────────────────────────────────────────────
  useEffect(() => {
    if (!handLandmarker) return;

    let animationFrameId;
    lastInferenceAtRef.current = 0;
    inFlightRef.current = false;

    const predict = () => {
      if (webcamRef.current && webcamRef.current.video.readyState === 4) {
        const video = webcamRef.current.video;
        const startTimeMs = performance.now();

        let results;
        try {
          results = handLandmarker.detectForVideo(video, startTimeMs);
        } catch (err) {
          animationFrameId = requestAnimationFrame(predict);
          return;
        }

        drawResults(results);

        const now = Date.now();
        if (!inFlightRef.current && now - lastInferenceAtRef.current >= INFERENCE_INTERVAL_MS) {
          lastInferenceAtRef.current = now;
          processDetection(results);
        }
      }
      animationFrameId = requestAnimationFrame(predict);
    };

    predict();
    return () => cancelAnimationFrame(animationFrameId);
  }, [handLandmarker, drawResults, processDetection]);

  return (
    <div className="flex flex-col gap-4 w-full h-full">
      <div className="relative rounded-2xl overflow-hidden bg-slate-900 shadow-2xl video-container border border-slate-800">
        <Webcam
          ref={webcamRef}
          mirrored={true}
          className="absolute inset-0 w-full h-full object-cover"
          videoConstraints={{
            width: 640,
            height: 480,
            facingMode: "user",
          }}
        />
        {/* Canvas WITHOUT scale-x-[-1] — we manually mirror x in drawResults */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover"
        />

        {isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm z-50">
            <div className="w-12 h-12 border-4 border-brand-cyan border-t-transparent rounded-full animate-spin mb-4"></div>
            <p className="text-slate-300 font-medium animate-pulse">Initializing AI Engine...</p>
          </div>
        )}

        {!prediction && !isLoading && (
          <div className="absolute top-6 left-1/2 -translate-x-1/2 z-10">
            <div className="px-5 py-2 bg-red-500/10 backdrop-blur-md border border-red-500/30 text-red-400 text-xs font-bold rounded-full uppercase tracking-widest shadow-lg flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
              {statusMessage}
            </div>
          </div>
        )}

        {prediction && (
          <div className="absolute top-6 left-1/2 -translate-x-1/2 z-10">
            <div className="px-6 py-3 bg-brand-cyan/20 backdrop-blur-md border border-brand-cyan/40 text-brand-cyan text-4xl font-black rounded-2xl shadow-2xl">
              {prediction}
            </div>
          </div>
        )}

        {!isLoading && handsDetected > 0 && (
          <div className="absolute bottom-4 left-4 z-10">
            <div className="px-3 py-1.5 bg-green-500/15 backdrop-blur-md border border-green-500/30 text-green-400 text-xs font-bold rounded-full flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400"></span>
              {handsDetected} Hand{handsDetected > 1 ? 's' : ''} Detected
            </div>
          </div>
        )}
      </div>

      {/* Confidence Bar */}
      <div className="p-6 bg-white/80 backdrop-blur-md rounded-2xl border border-slate-200 shadow-xl flex flex-col gap-4">
        <div className="flex justify-between items-end">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em]">Detection Confidence</span>
            <span className="text-3xl font-black font-mono text-slate-800 tracking-tighter">
              {confidence}%
            </span>
          </div>
          <div className="text-right">
             <span className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] block mb-1">Active Model</span>
             <span className="px-3 py-1 bg-slate-100 rounded-lg text-xs font-bold text-slate-600">{activeModel}</span>
          </div>
        </div>
        <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-brand-cyan to-blue-500 rounded-full transition-all duration-300 ease-out shadow-[0_0_12px_rgba(0,209,255,0.4)]"
            style={{ width: `${prediction ? confidence : 0}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default VideoPanel;
