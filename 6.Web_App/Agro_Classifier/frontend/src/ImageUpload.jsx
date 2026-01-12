import React, { useState } from "react";
import axios from "axios";
import "./Upload.css";

export default function ImageUpload({ crop }) {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const softLabelNames = {
    paddy: ["Sowing", "Vegetative", "Harvest"],
    tea: ["Plucked", "Flush", "Mature"],
    coconut: ["Low", "Medium", "High"],
  };

  const indicesNames = {
    paddy: ["NDVI", "EVI", "SAVI", "NDWI", "NDMI", "GNDVI"],
    tea: ["NDVI", "GNDVI", "EVI", "CIGREEN", "CIRE", "MSI"],
    coconut: ["NDVI", "EVI", "GNDVI", "SAVI", "NDMI", "MSI"],
  };

  const handleFileChange = (e) => setFiles(Array.from(e.target.files));

  const handleUpload = async () => {
    if (!crop || files.length === 0) return;
    setLoading(true);
    setResults([]);
    setError("");

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await axios.post(
        `http://localhost:8000/tiff/predict_${crop}_image`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResults(response.data);
    } catch (err) {
      console.error(err);
      setError("Upload failed. Check console.");
    } finally {
      setLoading(false);
    }
  };

  // Capitalize first letter of crop
  const cropName = crop.charAt(0).toUpperCase() + crop.slice(1);

  return (
    <div className="upload-container">
  {/* ------------------ Upload Box ------------------ */}
  <div className="upload-box">
    {/* Headline on top */}
    <div className="upload-headline">
      Upload {crop.charAt(0).toUpperCase() + crop.slice(1)} Images
    </div>

    {/* Upload Row */}
    <div className="upload-row">
      {/* Upload Icon */}
      <label htmlFor="file-upload" className="upload-icon">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="size-6"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0
               4.5 4.5M12 3v13.5"
          />
        </svg>
      </label>
      <input
        id="file-upload"
        type="file"
        multiple
        accept=".tif,.tiff"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {/* Selected Files & Clear */}
      {files.length > 0 && (
        <span className="selected-files">
          <i>{files.length} image{files.length > 1 ? "s" : ""} selected</i>
          <button type="button" onClick={() => setFiles([])} className="clear-btn">
            Clear
          </button>
        </span>
      )}

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={loading || files.length === 0}
        className="upload-btn"
      >
        {loading ? "Processing..." : "Classify"}
      </button>
    </div>
  </div>

  {error && <p className="error-text">{error}</p>}

  {/* Image Grid remains the same */}
  <div className="image-grid">
    {results.map((r, idx) => (
      <div key={idx} className="image-card">
        {r.image_base64 && <img src={r.image_base64} alt={`Image ${idx + 1}`} />}
        <p><b>Dominant Phase:</b> {r.dominant_phase || "-"}</p>
        {r.soft_labels && (
          <p>
            {r.soft_labels
              .map((v, i) => `${softLabelNames[crop][i]}: ${(v * 100).toFixed(1)}%`)
              .join(", ")}
          </p>
        )}
        {r.indices && (
          <p>
            {indicesNames[crop]
              .map((name) => `${name}: ${r.indices[name]?.toFixed(2) || "-"}`)
              .join(", ")}
          </p>
        )}
        {r.productivity_color && (
          <div className="productivity-section">
            <div
              className="productivity-bar"
              style={{ backgroundColor: `rgb(${r.productivity_color.join(",")})` }}
            ></div>
            <p>
              <b>Productivity:</b> {r.productivity_score?.toFixed(1)} ({r.productivity_level})
            </p>
          </div>
        )}
        {r.error && <p style={{ color: "red" }}>{r.error}</p>}
      </div>
    ))}
  </div>
</div>

  );
}
