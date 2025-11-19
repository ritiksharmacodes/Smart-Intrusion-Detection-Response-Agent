import { useEffect, useState } from "react";

export default function useFPSCounter() {
  const [fps, setFPS] = useState(0);

  useEffect(() => {
    let lastFrameTime = performance.now();
    let frameCount = 0;

    const update = () => {
      const now = performance.now();
      frameCount++;

      if (now - lastFrameTime >= 1000) {
        setFPS(frameCount);
        frameCount = 0;
        lastFrameTime = now;
      }

      requestAnimationFrame(update);
    };

    update();
  }, []);

  return fps;
}
