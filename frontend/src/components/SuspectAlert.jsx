function SuspectAlert({ suspect }) {
  // suspect = { name: "Rohit", score: 0.92 }
  if (!suspect) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: 10,
        right: 10,
        zIndex: 9999,
        background: "rgba(255, 0, 0, 0.85)",
        color: "white",
        padding: "12px 20px",
        borderRadius: "10px",
        fontSize: "14px",
        boxShadow: "0 0 15px rgba(255, 0, 0, 0.7)",
      }}
    >
      <div style={{ fontWeight: "bold", fontSize: "16px" }}>
        🚨 Suspect Detected
      </div>
      <div>Name: {suspect.name}</div>
      <div>Match Score: {(suspect.score * 100).toFixed(1)}%</div>
    </div>
  );
}

export default SuspectAlert;
