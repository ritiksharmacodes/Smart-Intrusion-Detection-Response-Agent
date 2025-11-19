// This canvas will be used for a semi-transparent heatmap.

import { useEffect, useRef } from "react";

function HeatmapOverlay({ videoRef }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video) return;

    const updateSize = () => {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    };

    updateSize();
    window.addEventListener("resize", updateSize);

    return () => window.removeEventListener("resize", updateSize);
  }, [videoRef]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        zIndex: 3,
        pointerEvents: "none",
        opacity: 0.6,
        border: "35px solid rgba(0,255,0,0.3)" // TEMP: to see canvas
      }}
    />
  );
}

export default HeatmapOverlay;
