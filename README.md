Detecting and Monitoring Agricultural Lands in Sri Lanka using Remote Sensing and Deep Learning
Project Overview

Agriculture is the cornerstone of Sri Lanka’s economy, yet traditional monitoring relies on manual field inspections and survey-based reporting that often lack timely, large-scale insights. This project introduces an automated, data-centric framework that integrates Remote Sensing, Deep Learning, and Web-based Visualization to provide real-time monitoring for three dominant crops: Paddy, Tea, and Coconut.

By leveraging harmonized Sentinel-2 satellite imagery and custom Convolutional Neural Networks (CNN), the system predicts crop growth stages and calculates a comprehensive productivity score. Unlike traditional classification models, this system utilizes a soft-labeling approach to capture the gradual transitions between phenological phases (e.g., from vegetative to harvest), reflecting biological reality more accurately than rigid hard-labeling.

Key Features

    Multi-Crop Support: Specialized monitoring for Paddy, Tea, and Coconut across diverse agro-ecological zones.

Soft-Label CNN Architecture: Captures continuous crop growth transitions and reduces misclassification errors caused by overlapping growth phases.

Weighted Productivity Scoring: Integrates CNN predictions (~70-75% weight) with biophysical spectral indices (~25-30% weight) to generate a normalized score (0-100).

Real-Time Visualization: Interactive dashboards and color-coded productivity maps (Low, Medium, High) for farmers and policymakers.

Geospatial Flexibility: Supports region-of-interest (ROI) queries via KML files, latitude/longitude coordinates, or direct image uploads.

System Architecture

The system is built on a scalable, full-stack architecture designed for high-performance geospatial processing:

    Frontend: Developed with React for interactive map-based visualizations using GeoJSON/KML overlays.

Backend: Powered by FastAPI to manage high-speed asynchronous model inference and image preprocessing.

Machine Learning Pipeline: Utilizes TensorFlow/Keras for patch-based CNN models.

Data Processing: Integrates Google Earth Engine (GEE) for acquiring harmonized Sentinel-2 surface reflectance data.

Database: MongoDB Atlas for storing time-series predictions, vegetation indices, and historical metadata.

Technical Methodology
1. Data Collection & Preprocessing

The dataset consists of 11,628 multi-temporal image patches collected from 2018 to 2025. Preprocessing includes:

    Cloud & Shadow Masking: Using the Scene Classification Layer (SCL) to ensure data reliability.

Band Normalization: Standardizing spectral bands (B2, B3, B4, B5, B8, B11) for consistent model input.

Patch Extraction: Images are divided into overlapping 64×64 pixel patches to capture local spatial context.

2. Crop-Specific Feature Extraction

The system computes specialized vegetation and water indices tailored to the physiological traits of each crop:

    Paddy: NDVI, EVI, GNDVI, SAVI, NDWI, NDMI (water-sensitive indices for flooded fields).

Tea: NDVI, GNDVI, EVI, CIgreen, CIred-edge, MSI (monitoring plucking cycles and chlorophyll).

Coconut: NDVI, EVI, GNDVI, SAVI, NDMI, MSI (reflecting canopy density and moisture).

3. CNN Training & Soft-Labeling

Instead of forcing a single classification, the models output a probability distribution across growth stages.

    Paddy Stages: Sowing, Vegetative, Harvest.

Tea Stages: Plucked, Flush, Mature.

Coconut Density: Low, Medium, High.

Results and Performance

The framework demonstrates high reliability across all target crops, with the stable canopy features of coconut leading to the highest precision.

Crop	Test Patches	Accuracy	F1-Score
Coconut	1,744	91.66%	0.9163
Tea	4,016	87.54%	0.8747
Paddy	5,868	86.36%	0.8617

(Data derived from system validation results.)

Productivity scores are normalized using real-world Sri Lankan yield benchmarks:

    Paddy: ∼4.2 t/ha

    Tea: ∼1.5 t/ha

    Coconut: ∼4.7 t/ha

Research Impact

This project bridges the gap between fragmented academic research and practical agricultural management. By providing timely data, it empowers stakeholders to proactively respond to climate variability, fertilizer shortages, and economic crises, ultimately supporting sustainable food security in Sri Lanka.
