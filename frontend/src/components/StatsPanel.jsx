import useFPSCounter from "../hooks/useFPSCounter";

function StatsPanel({ peopleCount = 0, threatScore = 0, connectionStatus = "closed" }) {
  const fps = useFPSCounter();

  const statusColor = connectionStatus === "open" ? "#0f0" : connectionStatus === "connecting" ? "#ff0" : "#f33";

  return (
    <div
      style={{
        position: "absolute",
        top: 10,
        left: 10,
        zIndex: 9999,
        background: "rgba(0,0,0,0.6)",
        padding: "10px 15px",
        color: "white",
        borderRadius: "8px",
        fontSize: "14px",
        lineHeight: "20px",
      }}
    >
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <div style={{
          width: 10, height: 10, borderRadius: 6, background: statusColor
        }} />
        <div style={{ fontSize: 12, opacity: 0.9 }}>{connectionStatus}</div>
      </div>

      <div style={{ marginTop: 8 }}>👥 People: {peopleCount}</div>
      <div>🎞 FPS: {fps}</div>
      <div>🔥 Threat Score: {threatScore}</div>
    </div>
  );
}

export default StatsPanel;
