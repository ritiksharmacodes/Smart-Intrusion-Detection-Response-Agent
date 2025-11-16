import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
// import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

function App() {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [model, setModel] = useState(null);
  const [peopleCount, setPeopleCount] = useState(0);
  const [fps, setFps] = useState(0);

  const PHONE_STREAM_URL = "http://192.168.29.45:8080/video"; // <-- replace this!

  useEffect(() => {
    const loadModel = async () => {
      console.log("Loading COCO-SSD model...");
      const m = await cocoSsd.load();
      setModel(m);
      console.log("COCO-SSD model loaded");
    };

    loadModel();
  }, []);

  useEffect(() => {
    if (!model) return;

    const video = videoRef.current;
    video.src = PHONE_STREAM_URL;
    video.crossOrigin = "anonymous";

    video.onloadeddata = () => {
      runDetection();
    };
  }, [model]);

  const runDetection = async () => {
    let last = performance.now();

    const detect = async () => {
      if (!videoRef.current || !model) return;

      const now = performance.now();
      setFps(Math.round(1000 / (now - last)));
      last = now;

      const predictions = await model.detect(videoRef.current);

      const people = predictions.filter((p) => p.class === "person");
      setPeopleCount(people.length);

      drawBoxes(people);

      requestAnimationFrame(detect);
    };

    detect();
  };

  const drawBoxes = (detections) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const video = videoRef.current;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "lime";
    ctx.font = "16px Arial";
    ctx.fillStyle = "lime";

    detections.forEach((det) => {
      const [x, y, width, height] = det.bbox;
      ctx.strokeRect(x, y, width, height);
      ctx.fillText(det.class + " " + Math.round(det.score * 100) + "%", x, y > 10 ? y - 5 : 10);
    });
  };

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
      <video
  ref={videoRef}
  id="phoneVideo"
  autoPlay
  muted
  playsInline
  controls={false}
  src={PHONE_STREAM_URL}
  crossOrigin="anonymous"
  style={{ width: "100%", height: "100%", background: "black" }}
/>


      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
        }}
      />

      <div
        style={{
          position: "absolute",
          top: 10,
          left: 10,
          background: "black",
          color: "white",
          padding: "10px",
          borderRadius: "8px",
        }}
      >
        People: {peopleCount} <br />
        FPS: {fps}
      </div>
    </div>
  );
}

export default App;
