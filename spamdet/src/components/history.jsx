function History({ messages }) {
  

  if (!messages || messages.length === 0) {
    return <p style={{ color: "gray" }}>⚠️ No history yet</p>;
  }

  return (
    <div
      style={{
        color:"black",
        border: "1px solid #ccc",
        padding: "15px",
        borderRadius: "5px",
        marginBottom: "20px",
        backgroundColor: "#f0f0f0",
      }}
    >
      <h2>History</h2>
      <ul>
        {messages.map((msg, index) => (
          <li key={index}>
            <strong>Prediction:</strong> {msg.prediction} <br />
            <strong>Text:</strong> {msg.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default History;
