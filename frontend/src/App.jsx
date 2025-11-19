// src/App.jsx (integration snippet)
import { useRef, useState, useEffect } from "react";
import VideoFeed from "./components/VideoFeed";
import CanvasOverlay from "./components/CanvasOverlay";
import HeatmapOverlay from "./components/HeatmapOverlay";
import StatsPanel from "./components/StatsPanel";
import EventLog from "./components/EventLog";
import SuspectAlert from "./components/SuspectAlert";

import useFrameCapture from "./hooks/useFrameCapture";
import useWebSocket from "./hooks/useWebSocket";

function App() {
  const mediaRef = useRef(null); // ref for <img> camera feed
  const [events, setEvents] = useState([]);
  const [suspect, setSuspect] = useState(null);
  const [peopleCount, setPeopleCount] = useState(0);
  const [threatScore, setThreatScore] = useState(0);

  // WebSocket (connects but Day1 server likely not present yet)
  const wsUrl = "ws://localhost:8000/ws/stream"; // change if needed
  const { status: connectionStatus, send } = useWebSocket(wsUrl, {
    onMessage: (msg) => {
      // Example backend messages parser
      if (msg?.event === "detection") {
        setPeopleCount(msg.people_count ?? 0);
        // optional: add event log entries
      } else if (msg?.event === "suspect_detected") {
        setSuspect({ name: msg.name, score: msg.score });
        setEvents((s) => [`Suspect detected: ${msg.name} (${Math.round(msg.score * 100)}%)`, ...s].slice(0, 50));
      } else if (msg?.event) {
        setEvents((s) => [JSON.stringify(msg), ...s].slice(0, 50));
      }
    }
  });

  // Frame capture (no canvas, direct JPEG fetch)
  const { start, stop } = useFrameCapture({
    intervalMs: 1000, // capture every second for Day 1
    onFrame: (blob, objectUrl) => {
      console.log("Captured JPEG frame (bytes):", blob.size);

      // For Day 1, do not send full frames yet
      if (connectionStatus === "open") {
        send({
          event: "frame_ready",
          size: blob.size,
        });
      }
    },
  });

  // Start capture when media is ready and WS is connected (optional)
  useEffect(() => {
    // For Day1 we don't auto-send, but we can start local capture to test pipeline:
    start();
    return () => stop();
  }, [start, stop]);

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative", overflow: "hidden" }}>
      <VideoFeed ref={mediaRef} />

      <CanvasOverlay videoRef={mediaRef} />
      <HeatmapOverlay videoRef={mediaRef} />

      <StatsPanel peopleCount={peopleCount} threatScore={threatScore} connectionStatus={connectionStatus} />

      <EventLog events={events} />
      <SuspectAlert suspect={suspect} />
    </div>
  );
}

export default App;
