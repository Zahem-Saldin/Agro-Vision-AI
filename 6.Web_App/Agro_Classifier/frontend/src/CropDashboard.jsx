import React, { useState } from "react";
import "./Dashboard.css";

// Crop images
import CoconutImg from "./Images/Coconut.jpg";
import PaddyImg from "./Images/Paddy.jpg";
import TeaImg from "./Images/Tea.jpg";

// Crop-specific components
import Paddy_PredictionForm from "./Paddy_PredictionForm";
import Tea_PredictionForm from "./Tea_PredictionForm";
import Coconut_PredictionForm from "./Coconut_PredictionForm";

// Separate upload components
import PaddyUpload from "./components/PaddyUpload";
import TeaUpload from "./components/TeaUpload";
import CoconutUpload from "./components/CoconutUpload";

// Map components
import MapComponent from "./MapComponent";
import Map from "./Map";

export default function CropDashboard() {
  const [crop, setCrop] = useState("paddy");
  const [activeRightPanel, setActiveRightPanel] = useState("form");

  const crops = [
    {
      name: "paddy",
      img: PaddyImg,
      formComponent: <Paddy_PredictionForm />,
      uploadComponent: <PaddyUpload />,
    },
    {
      name: "tea",
      img: TeaImg,
      formComponent: <Tea_PredictionForm />,
      uploadComponent: <TeaUpload />,
    },
    {
      name: "coconut",
      img: CoconutImg,
      formComponent: <Coconut_PredictionForm />,
      uploadComponent: <CoconutUpload />,
    },
  ];

  return (
    <div className="dashboard-container">
      <div className="top-grid">
        {/* Left Panel */}
        <div className="left-panel">
          <h2>Actions</h2>
          <div className="action-buttons">
            <button
              className={`action-btn ${activeRightPanel === "upload" ? "active" : ""}`}
              onClick={() => setActiveRightPanel("upload")}
            >
              File Upload & Classification
            </button>
            <button
              className={`action-btn ${activeRightPanel === "form" ? "active" : ""}`}
              onClick={() => setActiveRightPanel("form")}
            >
              KML Classification
            </button>
            <button
              className={`action-btn ${activeRightPanel === "map" ? "active" : ""}`}
              onClick={() => setActiveRightPanel("map")}
            >
              Polygon Map
            </button>
            <button
              className={`action-btn ${activeRightPanel === "map2" ? "active" : ""}`}
              onClick={() => setActiveRightPanel("map2")}
            >
              Show Map
            </button>
          </div>

          <h2>Select Crop Type</h2>
          <div className="crop-grid">
            {crops.map((c) => (
              <div
                key={c.name}
                className={`crop-card ${crop === c.name ? "selected" : ""}`}
                onClick={() => setCrop(c.name)}
              >
                <div className="crop-img-wrapper">
                  <img src={c.img} alt={c.name} className="crop-img" />
                  <span className="crop-label">
                    {c.name.charAt(0).toUpperCase() + c.name.slice(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Panel */}
        <div className="right-panel">
          {crops.map((c) => (
            <div key={c.name}>
              {/* Upload Panel */}
              <div
                style={{
                  display: activeRightPanel === "upload" && crop === c.name ? "block" : "none",
                }}
              >
                {c.uploadComponent}
              </div>

              {/* Form Panel */}
              <div
                style={{
                  display: activeRightPanel === "form" && crop === c.name ? "block" : "none",
                }}
              >
                {c.formComponent}
              </div>
            </div>
          ))}

          {/* MapComponent Panel */}
          <div
            style={{
              display: activeRightPanel === "map" ? "block" : "none",
              height: "100%",
            }}
          >
            <MapComponent crop={crop} visible={activeRightPanel === "map"} />
          </div>

          {/* Map.jsx Panel */}
          <div
            style={{
              display: activeRightPanel === "map2" ? "block" : "none",
              height: "100%",
            }}
          >
            <Map crop={crop} visible={activeRightPanel === "map2"} />
          </div>
        </div>
      </div>
    </div>
  );
}
