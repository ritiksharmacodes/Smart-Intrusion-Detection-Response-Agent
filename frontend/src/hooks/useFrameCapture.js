import { CAPTURE_URL } from "../utils/config";

export default function useFrameCapture({ intervalMs = 1000, onFrame = () => {} } = {}) {
  let timer = null;

  const captureOnce = async () => {
    try {
      const response = await fetch(CAPTURE_URL, {
        method: "GET",
        cache: "no-cache",
        mode: "no-cors",
      });

      const blob = await response.blob();

      // create objectURL for frontend use
      const url = URL.createObjectURL(blob);

      onFrame(blob, url);

      setTimeout(() => URL.revokeObjectURL(url), 500);

    } catch (err) {
      console.error("❌ Frame capture failed:", err);
    }
  };

  const start = () => {
    if (timer) return;
    timer = setInterval(captureOnce, intervalMs);
  };

  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
  };

  return { start, stop };
}
