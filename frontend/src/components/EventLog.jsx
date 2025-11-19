function EventLog({ events = [] }) {
  return (
    <div
      style={{
        position: "absolute",
        bottom: 10,
        left: 10,
        width: "280px",
        height: "200px",
        zIndex: 9999,
        background: "rgba(0,0,0,0.6)",
        color: "white",
        padding: "10px",
        borderRadius: "8px",
        fontSize: "12px",
        overflowY: "auto",
      }}
    >
      <div style={{ fontWeight: "bold", marginBottom: "6px" }}>
        Event Log
      </div>

      {events.length === 0 && (
        <div style={{ opacity: 0.5 }}>No events yet...</div>
      )}

      {events.map((e, index) => (
        <div key={index} style={{ marginBottom: "4px" }}>
          • {e}
        </div>
      ))}
    </div>
  );
}

export default EventLog;
