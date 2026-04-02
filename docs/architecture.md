# HydroVision Architecture

HydroVision is a Flask-powered ecosystem composed of modular microservices bridging geographical frontend requests to deep backend analytical engines.

## App Layout
1. **Frontend (`static/`)**: Relies purely on Vanilla.js and CSS modules. It uses Leaflet for mapping, and complex logic handles asynchronous polling, dynamic card building, and validation caching.
2. **Backend (`app.py` & `services/`)**: Flask exposes APIs which deserialize constraints (like date ranges, features, anomaly tolerances) and hands them instantly over to the individual services (`ChartService`, `AnalysisService`, `NetworkService`, etc.).
3. **Data Mesh (`data_loader.py`)**: The `DataRepository` lazily tracks and memoizes over 900+ csv files bridging disjointed meta-schemas (LamaH vs Mekong) into one unified API payload.

## Trustworthy Data vs. Heuristic Inference

Due to the complex nature of hydrological modeling, it is vital to know which parts of HydroVision emit absolute deterministic truth and which parts are heuristic estimates.

### 🟢 Trustworthy (Deterministic Methods)
These modules execute hard mathematics directly against physical, imported measurements. The outputs presented here are exact representations of reality.
- **`chart_service.py`**: Raw visualizations (Heatmaps, Plotly series, Year-over-Year aggregations). Missing data remains missing or is openly linearly interpolated.
- **`decomposition_service.py` & `wavelet_service.py`**: Provide STL mapping and exact frequency phase-transformations representing structural mathematical phenomenon inside the data scope.
- **`index_service.py` & `extreme_service.py`**: Extract real probabilistic thresholds from physical historical statistics (Standardised Precipitation Index, Return-Period Analysis).

### 🟡 Heuristic (Probability & External ML)
These models apply probability matrices, assumptions, AI models, and deep learning algorithms to fabricate inferences that *could* exist.
- **`scenario_service.py`**: Reconstructs simulated, hypothetical timeseries logic over target features by scaling variations applied to parent driver features.
- **`network_service.py`**: Utilizes Granger Causality matrices and lag mechanics to hallucinate probable relational correlations between disparate river systems. This defines "likely causality", not provable geographic causality.
- **`prediction_service.py`**: Forecasts data based off deep-learning tensors heavily influenced by its training phase and its rolling observation windows. Predictive uncertainty bands are strictly approximations.
- **Gemini NLP Automation**: When executing "Narrative Analysis" inside the Analysis Dock, an LLM observes the JSON metrics and fabricates a human-sounding report. While heavily anchored to the provided context, semantic subtleties could drift.
