import { useState } from "react";

function InputForm({ onSubmit }) {
  const [text, setText] = useState("");

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onSubmit(text);  // Send the text to parent component
      setText("");     // Clear the textarea after submission
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        marginBottom: "20px",
      }}
    >
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter a message to check..."
        style={{
          padding: "10px",
          fontSize: "16px",
          minHeight: "100px",
          borderRadius: "5px",
          border: "1px solid #ccc",
        }}
      />
      <button
        type="submit"
        style={{
          padding: "10px",
          fontSize: "16px",
          cursor: "pointer",
          borderRadius: "5px",
          border: "none",
          backgroundColor: "#0c6acfff",
          color: "white",
          
        }}
      >
        Check Spam
      </button>
    </form>
  );
}

export default InputForm;
