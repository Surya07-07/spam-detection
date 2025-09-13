function ResultDisplay({ result }) {
 

  if (!result) {
    return <p style={{ color: "gray" }}> No result yet</p>;
  }

  return (
    <div
      style={{
        color:"black",
        border: "1px solid #ccc",
        padding: "15px",
        borderRadius: "5px",
        marginBottom: "20px",
        backgroundColor: "#ffffffff",
      }}
    >
      <h2 style={{ marginBottom: "10px" }}>Prediction Result</h2>
      <p>
        <strong>Prediction:</strong> {result.prediction || " Missing prediction"}
      </p>
      <p>
        <strong>Text:</strong> {result.text || " Missing text"}
      </p>
    </div>
  );
}

export default ResultDisplay;
