import React, { useRef, useState, useEffect } from "react";
import { MapContainer, TileLayer, FeatureGroup, useMap } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";
import tokml from "tokml";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import "./MapComponent.css";
import Swal from "sweetalert2";
import withReactContent from "sweetalert2-react-content";

const MySwal = withReactContent(Swal);

const ResizeMap = ({ visible }) => {
  const map = useMap();
  useEffect(() => {
    if (visible) {
      const timer = setTimeout(() => map.invalidateSize({ animate: true }), 100);
      return () => clearTimeout(timer);
    }
  }, [map, visible]);
  return null;
};

const MapComponent = ({ visible = true }) => {
  const [geoJson, setGeoJson] = useState(null);
  const featureGroupRef = useRef(null);

  const TILE_INFO = {
    center: [7.8731, 80.7718],
    minZoom: 8,
    maxZoom: 15,
    bounds: [
      [5.9, 79.5],
      [9.8, 81.9],
    ],
    tileExtension: "png",
  };

  const updateGeoJson = () => {
    if (featureGroupRef.current) {
      const drawnItems = featureGroupRef.current;
      const geojsonData = drawnItems.toGeoJSON();
      setGeoJson(geojsonData);
    }
  };

  const copyKML = () => {
    if (!geoJson || !geoJson.features.length) {
      MySwal.fire({
        icon: "warning",
        title: "No shapes drawn!",
      });
      return;
    }
    const kml = tokml(geoJson);

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(kml)
        .then(() => {
          MySwal.fire({
            icon: "success",
            title: "KML copied!",
            timer: 1500,
            showConfirmButton: false,
          });
        })
        .catch(() => fallbackCopy(kml));
    } else {
      fallbackCopy(kml);
    }
  };

  const fallbackCopy = (text) => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand("copy");
      MySwal.fire({
        icon: "success",
        title: "KML copied (fallback)!",
        timer: 1500,
        showConfirmButton: false,
      });
    } catch (err) {
      console.error("Unable to copy KML:", err);
      MySwal.fire({
        icon: "error",
        title: "Failed to copy KML",
        text: "Check console for details",
      });
    }
    document.body.removeChild(textarea);
  };

  return (
    <div className="map-wrapper">
      <MapContainer
        center={TILE_INFO.center}
        zoom={TILE_INFO.minZoom}
        minZoom={TILE_INFO.minZoom}
        maxZoom={TILE_INFO.maxZoom}
        maxBounds={TILE_INFO.bounds}
        style={{ height: "100%", width: "100%" }}
      >
        <ResizeMap visible={visible} />

        {/* Sri Lanka GDAL tiles */}
        <TileLayer
          url={`/tile/{z}/{x}/{y}.${TILE_INFO.tileExtension}`}
          attribution="&copy; Your Tiles"
          noWrap={true}
          bounds={TILE_INFO.bounds}
        />

        {/* Labels */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          attribution="Labels © Esri"
          pane="overlayPane"
          bounds={TILE_INFO.bounds}
        />

        {/* Drawing Tools */}
        <FeatureGroup ref={featureGroupRef}>
          <EditControl
            position="topleft"
            draw={{
              polygon: true,
              marker: true,
              rectangle: false,
              polyline: false,
              circle: false,
              circlemarker: false,
            }}
            edit={{ remove: true }}
            onCreated={updateGeoJson}
            onEdited={updateGeoJson}
            onDeleted={updateGeoJson}
          />
        </FeatureGroup>
      </MapContainer>

      {/* Copy KML button */}
      <button className="copy-kml-btn" onClick={copyKML}>
        Copy KML
      </button>
    </div>
  );
};

export default MapComponent;
