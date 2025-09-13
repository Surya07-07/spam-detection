function FrontPage({ onStart }) {
  return (
    
    <div 
      style={{
        margin:"0px",
        width:"100vw",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        background: "linear-gradient(to right, #4facfe, #00f2fe)",
        color: "white",
        textAlign: "center",
      }}
    >
      <h1 style={{ fontSize: "3rem", marginBottom: "20px" }}>
        🚀 Spam Detection App
      </h1>
      <p style={{ fontSize: "1.2rem", maxWidth: "600px", marginBottom: "30px" }}>
        Check if your emails or messages are spam instantly using AI-powered
        detection.
      </p>
      <button
        onClick={onStart}
        style={{
          padding: "15px 30px",
          fontSize: "18px",
          borderRadius: "10px",
          border: "none",
          backgroundColor: "#007BFF",
          color: "white",
          cursor: "pointer",
        }}
      >
        Get Started
      </button>
    </div>
  );
}

export default FrontPage;
