import React, { useState } from "react";
import axios from "axios";

const App = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPrediction(response.data.prediction);
      setConfidence(`Confidence: ${(response.data.confidence * 100).toFixed(2)}%`);
    } catch (error) {
      console.error("Error making prediction:", error);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Diabetic Retinopathy Detection</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Predict</button>
      {prediction && (
        <div>
          <h2>Prediction: {prediction}</h2>
          <p>{confidence}</p>
        </div>
      )}
    </div>
  );
};

export default App;
