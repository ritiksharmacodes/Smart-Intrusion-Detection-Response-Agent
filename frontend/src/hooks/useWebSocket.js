// src/hooks/useWebSocket.js
import { useEffect, useRef, useState, useCallback } from "react";

/**
 * useWebSocket(url, { onMessage })
 * returns { status, send, close }
 * status: "connecting" | "open" | "closed" | "error"
 */
export default function useWebSocket(url, { onMessage = () => {} } = {}) {
  const socketRef = useRef(null);
  const [status, setStatus] = useState("closed");

  useEffect(() => {
    if (!url) return;

    let ws;
    try {
      setStatus("connecting");
      ws = new WebSocket(url);
      socketRef.current = ws;
    } catch (err) {
      setStatus("error");
      console.error("WebSocket create error", err);
      return;
    }

    ws.onopen = () => {
      setStatus("open");
      console.log("[ws] open", url);
    };

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        onMessage(data);
      } catch (e) {
        // If message is binary or non-JSON, pass raw event
        onMessage(ev.data);
      }
    };

    ws.onerror = (err) => {
      console.error("[ws] error", err);
      setStatus("error");
    };

    ws.onclose = (ev) => {
      console.log("[ws] closed", ev);
      setStatus("closed");
    };

    return () => {
      try { ws.close(); } catch {}
      socketRef.current = null;
    };
  }, [url, onMessage]);

  const send = useCallback((obj) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      console.warn("[ws] send failed, socket not open");
      return false;
    }
    try {
      const text = typeof obj === "string" ? obj : JSON.stringify(obj);
      socketRef.current.send(text);
      return true;
    } catch (e) {
      console.error("[ws] send error", e);
      return false;
    }
  }, []);

  const close = useCallback(() => {
    if (socketRef.current) socketRef.current.close();
  }, []);

  return { status, send, close, socketRef };
}
