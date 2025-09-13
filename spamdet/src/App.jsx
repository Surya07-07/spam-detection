import { useState } from "react";
import InputForm from "./components/InputForm";
import ResultDisplay from "./components/ResultDisplay";
import History from "./components/history";
import FrontPage from "./components/iogin"

function App() {
  const [started, setStarted] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const handleCheckSpam = async (text) => {
    

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

     

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      

      setResult(data);
      

      setHistory((prev) => [data, ...prev]);
      

    } catch (error) {
      console.error(" Error fetching prediction:", error);
      alert("Failed to get prediction. Check backend is running!");
    }
  };
if (!started) {
    return <FrontPage onStart={() => setStarted(true)} />;
  }
  

  return (
    <div style={{ width: "800px",minHeight:"100vh", margin: "40px", fontFamily: "Arial, sans-serif",marginTop:"80px",overflow:"hidden" }}>
      <h1 style={{ color:"lightblue",textAlign: "center", marginBottom: "30px" }}>Spam Detection App</h1>

      <InputForm onSubmit={handleCheckSpam} />
      <ResultDisplay result={result} />
      <History messages={history} />
    </div>
  );
}

export default App;
