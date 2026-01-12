import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Polygon, Popup, useMap } from "react-leaflet";
import axios from "axios";
import Swal from "sweetalert2";
import "leaflet/dist/leaflet.css";
import "./CropMap.css";

// ---------- Helper: parse polygons from KML ----------
const parseKML = (kml_text) => {
  const polygons = [];
  try {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(kml_text, "text/xml");
    const placemarks = xmlDoc.getElementsByTagName("Placemark");

    for (let i = 0; i < placemarks.length; i++) {
      const polygon = placemarks[i].getElementsByTagName("Polygon")[0];
      if (!polygon) continue;

      const coordsText = polygon.getElementsByTagName("coordinates")[0].textContent;
      const coords = coordsText
        .trim()
        .split(/\s+/)
        .map((c) => {
          const [lng, lat] = c.split(",").map(Number);
          return [lat, lng]; // Leaflet uses [lat, lng]
        });

      polygons.push(coords);
    }
  } catch (err) {
    console.error("Failed to parse KML:", err);
  }
  return polygons;
};

// Centering helper for Leaflet inside flex container
function RecenterMap({ center }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center);
    map.invalidateSize(); // Important for flex layout
  }, [center, map]);
  return null;
}

// ---------- Phase names for each crop ----------
const PHASE_NAMES = {
  paddy: ["Sowing", "Vegetative", "Harvest"],
  tea: ["Plucked", "Flush", "Mature"],
  coconut: ["Low", "Medium", "High"],
};

export default function CropMap({ crop }) {
  const [data, setData] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [year, setYear] = useState(2020);
  const [quarter, setQuarter] = useState(1);
  const [mapType, setMapType] = useState("osm");
  const [loading, setLoading] = useState(false);

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

  const mapCenter = mapType === "osm" ? [7.515, 80.989] : TILE_INFO.center;
  const phaseNames = PHASE_NAMES[crop] || [];

  // ---------- Fetch crop data ----------
  const fetchData = async () => {
    setLoading(true);
    try {
      const url = `http://127.0.0.1:8000/save/predictions_${crop}`;
      const res = await axios.get(url);

      if (res.data.status === "success") setData(res.data.data);
      else setData([]);
    } catch (err) {
      console.error(err);
      setData([]);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch on mount
  useEffect(() => {
    fetchData();
  }, [crop]);

  // ---------- Refresh button ----------
  const handleRefresh = () => {
    fetchData();
  };

  // ---------- Delete polygon ----------
  const handleDelete = async () => {
    if (!selectedId) return;
    const result = await Swal.fire({
      title: "Are you sure?",
      text: "This will delete the selected polygon and its predictions!",
      icon: "warning",
      showCancelButton: true,
      confirmButtonColor: "#d33",
      cancelButtonColor: "#3085d6",
      confirmButtonText: "Yes, delete it!",
    });

    if (result.isConfirmed) {
      try {
        await axios.delete(`http://127.0.0.1:8000/save/predictions_${crop}/${selectedId}`);
        Swal.fire("Deleted!", "The polygon has been deleted.", "success");
        setSelectedId(null);
        fetchData(); // refresh data immediately after deletion
      } catch (err) {
        console.error(err);
        Swal.fire("Error", "Failed to delete polygon", "error");
      }
    }
  };

  return (
    <div className="cropmap-container">
      {/* Controls */}
      <div className="controls">
        <label>
          Year:
          <input
            type="number"
            value={year}
            onChange={(e) => setYear(parseInt(e.target.value))}
          />
        </label>

        <label>
          Quarter:
          <input
            type="number"
            min={1}
            max={4}
            value={quarter}
            onChange={(e) => setQuarter(parseInt(e.target.value))}
          />
        </label>

        <label>
          Map:
          <select value={mapType} onChange={(e) => setMapType(e.target.value)}>
            <option value="osm">Street Map</option>
            <option value="custom">3D Map</option>
          </select>
        </label>

        <button className="refresh-btn" onClick={handleRefresh} title="Refresh">
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={2}
    stroke="green"
    className="size-6"
    style={{ width: "24px", height: "24px" }} // optional size styling
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"
    />
  </svg>
</button>

      </div>

      {/* Map + Cards */}
      <div className="dashboard-split">
        <MapContainer
          center={mapCenter}
          zoom={mapType === "osm" ? 13 : TILE_INFO.minZoom}
          minZoom={TILE_INFO.minZoom}
          maxZoom={TILE_INFO.maxZoom}
          maxBounds={TILE_INFO.bounds}
          className="map-panel"
        >
          <RecenterMap center={mapCenter} />

          {mapType === "osm" ? (
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="&copy; OpenStreetMap contributors"
            />
          ) : (
            <>
              <TileLayer
                url={`/tile/{z}/{x}/{y}.${TILE_INFO.tileExtension}`}
                attribution="&copy; Your Tiles"
                noWrap={true}
                bounds={TILE_INFO.bounds}
              />
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                attribution="Labels © Esri"
                pane="overlayPane"
                bounds={TILE_INFO.bounds}
              />
            </>
          )}

          {/* Polygons */}
          {data.map((item) => {
            const polygons = parseKML(item.kml_text);
            return polygons.map((coords, idx) => {
              const prediction =
                item.predictions.find(
                  (p) => p.year === year && p.quarter === quarter
                ) || { productivity_color: [0, 0, 0] };

              const color = `rgb(${prediction.productivity_color.join(",")})`;

              return (
                <Polygon
                  key={`${crop}-${item._id}-${idx}`}
                  positions={coords}
                  pathOptions={{
                    color: "black",
                    weight: 1,
                    fillColor: color,
                    fillOpacity: 0.5,
                  }}
                  eventHandlers={{
                    click: () => setSelectedId(item._id),
                  }}
                >
                  <Popup>
                    <div>
                      <strong>Year:</strong> {year}
                      <br />
                      <strong>Quarter:</strong> {quarter}
                      <br />
                      <strong>Phase:</strong> {prediction.dominant_phase || "-"}
                      <br />
                      <strong>Productivity Score:</strong>{" "}
                      {prediction.productivity_score?.toFixed(2) || 0}
                      <br />
                      <strong>Level:</strong>{" "}
                      {prediction.productivity_level || "-"}
                    </div>
                  </Popup>
                </Polygon>
              );
            });
          })}
        </MapContainer>

        {/* Right Panel */}
        <div className="cards-panel">
          <h3>
            <center>
              {crop.charAt(0).toUpperCase() + crop.slice(1)} Classification{" "}
              {selectedId ? `` : "(Click a polygon)"}
            </center>
          </h3>

          {selectedId && (
            <div className="cards-grid">
              {data
                .filter((item) => item._id === selectedId)
                .flatMap((item) => item.predictions)
                .filter((r) => r.year === year && r.quarter === quarter)
                .map((r, q) => (
                  <div key={`${r.year}-Q${r.quarter}-${q}`} className="quarter-card">
                    {r.image_base64 && (
                      <img
                        src={r.image_base64}
                        alt={`Year ${r.year} Q${r.quarter}`}
                      />
                    )}
                    <div className="card-details">
                      <p>
                        <b>Q{r.quarter} Dominant Phase:</b> {r.dominant_phase || "-"}
                      </p>
                      {r.soft_labels && (
                        <p>
                          {r.soft_labels
                            .map(
                              (v, i) =>
                                `${phaseNames[i] || "Phase " + (i + 1)}: ${
                                  (v * 100).toFixed(1)
                                }%`
                            )
                            .join(", ")}
                        </p>
                      )}
                      {r.indices && (
                        <div className="indices">
                          {Object.entries(r.indices).map(([key, value]) => (
                            <div key={key}>
                              <b>{key.toUpperCase()}:</b> {value?.toFixed(2) || "-"}
                            </div>
                          ))}
                        </div>
                      )}
                      {/* SHOW index_explanations if present */}
                        {r.index_explanations && Object.keys(r.index_explanations).length > 0 && (
                          <div className="index-explanations">
                            <h4>- - - - - - - - Index Explanations - - - - - - - - </h4>
                            {Object.entries(r.index_explanations).map(([idxKey, info]) => {
                              // info expected to be { value, effect, description }
                              const value = info?.value ?? (r.indices?.[idxKey.toLowerCase()] ?? null);
                              const effect = info?.effect ?? "";
                              const desc = info?.description ?? "";
                              return (
                                <div key={idxKey} className="explanation-row" style={{ textAlign: "left" }}>
  <b>{idxKey}</b>
  <div className="explanation-detail">
    <div><small><i>{desc}</i></small></div>

    <div>
      <small>
        <b>Interpretation:</b>{" "}
        <span
          style={{
            color: effect?.toLowerCase().includes("positive") ? "green" :
                   effect?.toLowerCase().includes("negative") ? "red" : "black",
            fontWeight: "bold"
          }}
        >
          {effect}
        </span>
      </small>
    </div>

    <div>
      <small>
        <b>Value:</b>{" "}
        {value !== null ? Number(value).toFixed(3) : "-"}
      </small>
    </div>
  </div>
</div>

                              );
                            })}
                          </div>
                        )}
                      {r.productivity_color && (
                        <div className="productivity-section">
                          <div
                            className="productivity-bar"
                            style={{
                              backgroundColor: `rgb(${r.productivity_color.join(",")})`,
                            }}
                          ></div>
                          <p>
                            <b>Productivity Level:</b> {r.productivity_level} (
                            {r.productivity_score?.toFixed(1)})
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

              <p>
                <button className="delete-btn" onClick={handleDelete}>
                  Delete Area
                </button>
              </p>
            </div>
          )}
        </div>
      </div>

      {loading && <div className="loading-overlay">Loading...</div>}
    </div>
  );
}
