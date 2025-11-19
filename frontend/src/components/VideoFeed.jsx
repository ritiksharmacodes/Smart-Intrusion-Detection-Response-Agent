import { forwardRef, useEffect, useState } from "react";
import { PHONE_STREAM_URL } from "../utils/config";

const VideoFeed = forwardRef((props, ref) => {
  const [imgSrc, setImgSrc] = useState("");

  useEffect(() => {
    let frame = 0;

    const update = () => {
      // Add unique query each time so browser reloads image
      setImgSrc(`${PHONE_STREAM_URL}?frame=${frame++}`);
    };

    const interval = setInterval(update, 100); // 10 FPS approximate
    update();

    return () => clearInterval(interval);
  }, []);

  return (
    <img
      ref={ref}
      src={imgSrc}
      alt="Camera Feed"
      style={{
        width: "100%",
        height: "100%",
        objectFit: "cover",
        position: "absolute",
        top: 0,
        left: 0,
        zIndex: 1,
      }}
    />
  );
});

export default VideoFeed;
