// This canvas will be used later to draw YOLO boxes via drawBoxes.js.

import { useEffect, useRef } from "react";

function CanvasOverlay({ videoRef }) {
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
        zIndex: 2,
        pointerEvents: "none",
        border: "100px solid rgba(255,0,0,0.3)" // TEMP: to make canvas visible
      }}
    />
  );
}

export default CanvasOverlay;
