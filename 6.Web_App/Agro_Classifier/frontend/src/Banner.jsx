import React from "react";
import CropDashboard from "./CropDashboard";
import "./Banner.css";

const Banner = () => {
  return (
    <div className="banner-container">
      <div className="banner-header">
        <h1 className="title">- Crop Growth Classification -</h1>
      </div>

      <CropDashboard />
    </div>
  );
};

export default Banner;
