import React, { useState, useEffect } from "react";
import "./PredictionForm.css";
import Swal from "sweetalert2";

export default function Coconut_PredictionForm() {
  const [kml, setKml] = useState("");
  const [fileLoaded, setFileLoaded] = useState(false);
  const [startYear, setStartYear] = useState();
  const [endYear, setEndYear] = useState();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const phaseNames = ["Low", "Medium", "High"];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResults([]);

    try {
      const response = await fetch("http://localhost:8000/kml/predict_coconut", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kml_text: kml,
          start_year: +startYear,
          end_year: +endYear,
        }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let { value, done } = await reader.read();
      let buffer = "";
      let isFirstChunk = true;

      while (!done) {
        buffer += decoder.decode(value, { stream: true });

        if (isFirstChunk) {
          buffer = buffer.replace(/^\s*\[/, "");
          isFirstChunk = false;
        }

        let parts = buffer.split(/(?<=\}),(?=\{)/);
        buffer = parts.pop();

        for (const part of parts) {
          if (!part.trim()) continue;
          try {
            setResults((prev) => [...prev, JSON.parse(part)]);
          } catch (err) {
            console.error("Failed to parse chunk:", part, err);
          }
        }

        ({ value, done } = await reader.read());
      }

      buffer = buffer.replace(/\]\s*$/, "").trim();
      if (buffer) {
        try {
          setResults((prev) => [...prev, JSON.parse(buffer)]);
        } catch (err) {
          console.error("Failed to parse final buffer:", buffer, err);
        }
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveToDb = async () => {
    try {
      const payload = {
        kml_text: kml,
        start_year: Number(startYear),
        end_year: Number(endYear),
        predictions: results.map((r) => ({
          year: Number(r.year) || 0,
          quarter: Number(r.quarter) || 0,
          dominant_phase: r.dominant_phase || "",
          soft_labels: Array.isArray(r.soft_labels)
            ? r.soft_labels.map((v) => Number(v))
            : [0, 0, 0],
          indices: r.indices || {},
          // <-- include index_explanations as-is (object with keys like NDVI -> {value, effect, description})
          index_explanations: r.index_explanations || {},
          image_base64: r.image_base64 || null,
          productivity_color: Array.isArray(r.productivity_color)
            ? r.productivity_color.map((v) => Number(v))
            : [0, 0, 0],
          productivity_score:
            r.productivity_score !== undefined ? Number(r.productivity_score) : 0,
          productivity_level: r.productivity_level || "-",
        })),
      };

      const response = await fetch("http://127.0.0.1:8000/save/predictions_coconut", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error(`Save failed: ${response.status}`);
      Swal.fire("Saved!", "Data saved successfully.", "success");
    } catch (err) {
      console.error("Save to DB error:", err);
      Swal.fire("Error", "Failed to save data.", "error");
    }
  };

  // Group results by year and quarter
  const groupedResults = results.reduce((acc, r) => {
    if (r.quarter) {
      if (!acc[r.year]) acc[r.year] = {};
      acc[r.year][r.quarter] = r;
    }
    return acc;
  }, {});

  const totalQuarters = (endYear - startYear + 1) * 4;
  const loadedQuarters = results.filter((r) => r.quarter).length;
  const progress = Math.min((loadedQuarters / totalQuarters) * 100, 100);

  // --- SweetAlert when loading reaches 100% ---
  useEffect(() => {
    if (!loading && progress === 100 && results.length > 0) {
      Swal.fire({
        title: "Prediction Complete!",
        text: "Do you want to save the results to the database?",
        icon: "success",
        showCancelButton: true,
        confirmButtonText: "Save",
        cancelButtonText: "Cancel",
      }).then((result) => {
        if (result.isConfirmed) {
          handleSaveToDb();
        }
      });
    }
  }, [loading, progress, results]);

  return (
    <div>
      <div className="prediction-grid">
        {/* Left column: Instructions */}
        <div className="instructions-column">
          <div>
            <h3>How to Use the Coconut Growth Prediction Tool</h3>
            <p>
              <b>Prerequisites:</b> You need a Google account and a Google Earth project.
            </p>
            <ol>
              <li>Open <b>Google Earth</b> on your browser.</li>
              <li>Search for the location of your crop land.</li>
              <li>
                Mark your crop area:
                <ul>
                  <li>Draw a polygon or path around the crop land, <b>or</b> add a placemark for a specific location.</li>
                </ul>
              </li>
              <li>Save your polygon, path, or placemark to your <b>Google Earth project</b>.</li>
              <li>Right-click the feature and choose <b>Copy</b>.</li>
              <li>Come back to this tool and click the <b>clipboard icon</b> to paste the KML feature here.</li>
            </ol>
            <p><i>After pasting, select Start Year → End Year → Click Predict to see predictions.</i></p>
          </div>
        </div>

        {/* Right column: Form */}
        <div className="form-column">
          <div className="prediction-form">
            <form onSubmit={handleSubmit}>
              {/* KML Input */}
              <label className="full-width">
                <center>
                  <i>Paste Google Earth KML:</i>
                  <br />

                  {!fileLoaded && (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={1.5}
                      stroke="currentColor"
                      className="icon-big"
                      style={{ cursor: "pointer", marginTop: "12px" }}
                      title="Paste KML from clipboard"
                      onClick={async () => {
                        try {
                          const text = await navigator.clipboard.readText();
                          if (text) {
                            setKml(text);
                            setFileLoaded(true);
                          } else {
                            alert("Clipboard is empty or does not contain KML.");
                          }
                        } catch (err) {
                          console.error("Clipboard read failed:", err);
                          alert(
                            "Failed to read clipboard. Make sure your browser allows clipboard access."
                          );
                        }
                      }}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 0 0 2.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 0 0-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75 2.25 2.25 0 0 0-.1-.664m-5.8 0A2.251 2.251 0 0 1 13.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25ZM6.75 12h.008v.008H6.75V12Zm0 3h.008v.008H6.75V15Zm0 3h.008v.008H6.75V18Z"
                      />
                    </svg>
                  )}

                  {fileLoaded && (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={1.5}
                      stroke="currentColor"
                      className="icon-big"
                      style={{ cursor: "pointer", marginTop: "12px" }}
                      title="Remove pasted KML"
                      onClick={() => {
                        setKml("");
                        setFileLoaded(false);
                      }}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="m9.75 9.75 4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
                      />
                    </svg>
                  )}

                  {fileLoaded && <p>Success!!</p>}
                </center>
              </label>

              {/* Year Inputs */}
              {fileLoaded && (
                <div className="year-inputs-grid">
                  <label>Start Year:</label>
                  <div className="year-grid">
                    {[2020, 2021, 2022, 2023, 2024].map((year) => (
                      <div
                        key={`start-${year}`}
                        className={`year-box ${startYear === year ? "selected" : ""}`}
                        onClick={() => {
                          setStartYear(year);
                          if (endYear && endYear < year) setEndYear(null);
                        }}
                      >
                        {year}
                      </div>
                    ))}
                  </div>

                  {startYear && (
                    <>
                      <label>End Year:</label>
                      <div className="year-grid">
                        {[2020, 2021, 2022, 2023, 2024].map((year) => (
                          <div
                            key={`end-${year}`}
                            className={`year-box ${endYear === year ? "selected" : ""} ${
                              year < startYear ? "disabled" : ""
                            }`}
                            onClick={() => {
                              if (year >= startYear) setEndYear(year);
                            }}
                          >
                            {year}
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              {fileLoaded && (
                <div className="predict-btn-container">
  <button
    type="submit"
    className="predict-btn"
    disabled={loading || !startYear || !endYear}
  >
    Classify
  </button>
</div>
              )}
            </form>
          </div>
        </div>
      </div>

      {loading && (
        <div className="full-width-div">
          <div className="progress-container">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          </div>
          <p className="loading-text">
            Loading... {loadedQuarters}/{totalQuarters} quarters ({progress.toFixed(0)}%)
          </p>
        </div>
      )}

      {error && <p className="error-text">Error: {error}</p>}

      {/* Results display (unchanged) */}
      {Object.entries(groupedResults).map(([year, yearResults]) => {
        const quarters = [1, 2, 3, 4].map((q) => yearResults[q]).filter(Boolean);
        let dynamicYearlySummary = null;
        let avgColor = [0, 0, 0];

        if (quarters.length > 0) {
          const avgSoftLabels = [0, 0, 0];
          let totalProductivityScore = 0;

          quarters.forEach((q) => {
            q.soft_labels?.forEach((v, i) => (avgSoftLabels[i] += v));
            if (q.productivity_score !== undefined)
              totalProductivityScore += q.productivity_score;
            if (q.productivity_color) {
              avgColor[0] += q.productivity_color[0];
              avgColor[1] += q.productivity_color[1];
              avgColor[2] += q.productivity_color[2];
            }
          });

          avgSoftLabels.forEach((v, i) => (avgSoftLabels[i] = v / quarters.length));
          const avgProductivityScore = totalProductivityScore / quarters.length || 0;
          avgColor = avgColor.map((v) => Math.round(v / quarters.length));

          const phaseCounts = { Low: 0, Medium: 0, High: 0 };
          quarters.forEach((q) => {
            if (q.dominant_phase) phaseCounts[q.dominant_phase] += 1;
          });

          const dominant_phase_percentage = {};
          Object.entries(phaseCounts).forEach(([phase, count]) => {
            dominant_phase_percentage[phase] = (count / quarters.length) * 100;
          });

          dynamicYearlySummary = {
            yearly_avg_soft_labels: avgSoftLabels,
            yearly_avg_productivity_score: avgProductivityScore,
            dominant_phase_percentage,
            yearly_productivity_level: quarters[quarters.length - 1]?.productivity_level || "-",
            yearly_avg_color: avgColor,
          };
        }

        return (
          <div key={year} className="results-year">
            {dynamicYearlySummary && (
              <div className="year-summary">
                <h2>Year {year}</h2>
                {dynamicYearlySummary.yearly_avg_color && (
                  <div className="productivity-section">
                    <div
                      className="productivity-bar"
                      style={{
                        backgroundColor: `rgb(${dynamicYearlySummary.yearly_avg_color.join(",")})`,
                      }}
                    ></div>
                  </div>
                )}
                <p>
                  <b>Yearly Avg Soft Labels:</b>{" "}
                  {dynamicYearlySummary.yearly_avg_soft_labels
                    .map((v, i) => `${phaseNames[i]}: ${(v * 100).toFixed(1)}%`)
                    .join(", ")}
                </p>
                <p>
                  <b>Yearly Avg Productivity Score:</b>{" "}
                  {dynamicYearlySummary.yearly_avg_productivity_score.toFixed(1)} (
                  {dynamicYearlySummary.yearly_productivity_level})
                </p>
                <p>
                  <b>Dominant Phase %:</b>{" "}
                  {Object.entries(
                    dynamicYearlySummary.dominant_phase_percentage
                  )
                    .map(([phase, pct]) => `${phase}: ${pct.toFixed(1)}%`)
                    .join(", ")}
                </p>

              </div>
            )}

            <div className="quarters-grid">
              {[1, 2, 3, 4].map((q) => {
                const r = yearResults[q];
                return (

                  <div key={q} className="quarter-card">
                    {!r ? (
                      loading ? (
                        <div className="spinner"></div>
                      ) : (
                        <p>No data</p>
                      )
                    ) : (

                      <div>

                        {r.image_base64 && (
                          <img
                            src={r.image_base64}
                            alt={`Year ${r.year} Q${r.quarter}`}
                          />
                        )}
                        <p>
                          <b>Q{r.quarter} Dominant Phase:</b>{" "}
                          {r.dominant_phase || "-"}
                        </p>
                        <p>
                          {r.soft_labels
                            ?.map(
                              (v, i) =>
                                `${phaseNames[i]}: ${(v * 100).toFixed(1)}%`
                            )
                            .join(", ")}
                        </p>
                        {r.indices && (
                          <div className="indices">
                            {Object.entries(r.indices).map(
                              ([key, value]) => (
                                <div key={key}>
                                  <b>{key.toUpperCase()}:</b>{" "}
                                  {value?.toFixed(2) || "-"}
                                </div>
                              )
                            )}
                          </div>
                        )}

                        {/* SHOW index_explanations if present */}
                        {r.index_explanations && Object.keys(r.index_explanations).length > 0 && (
                          <div className="index-explanations">
                            <h4>- - - - - - - - - Index Explanations - - - - - - - - - </h4>
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

                      </div>
                    )}

                  </div>
                );
              })}

            </div>


          </div>
        );
      })}
    </div>
  );
}
