# HydroVision

A modern, Flask-based spatial analytics dashboard built specifically for deep temporal exploration of hydrological monitoring stations.

**Live demo:** [https://hydrovision.onrender.com/](https://hydrovision.onrender.com/)

---

## Overview

HydroVision is a multi-dataset hydrological analysis engine combining deterministic mathematical models with AI-assisted narrative pipelines. It supports two large-scale datasets — the **Mekong River Basin** (Southeast Asia) and **LamaH-CE** (Central Europe) — and provides a unified interface for exploring discharge, rainfall, water level, temperature, and sediment data across hundreds of monitoring stations.

The platform is designed for researchers, students, and water resource practitioners who need to go beyond basic plotting and want to understand the structure, dynamics, and future trajectory of hydrological systems.

---

## What You Can Do

### Explore & Visualise
- Plot time series for any station and feature across both datasets
- Build a persistent **Visualisation Dock** of multiple comparison cards side-by-side
- Chart types include: Time Series, Flow Duration Curves, Seasonal Climatology, Monthly Heatmaps, Scatter plots, Dual-Axis overlays, Box plots, and more
- Apply date range filters and overlay multiple stations on a single graph

### Analyse
- **Free Analysis**: Select any series and receive AI-generated statistical narratives with automatic chart recommendations
- **Compare**: Directly compare two stations or features with correlation and lag detection
- **Network**: Explore river connectivity, travel-time lags between upstream/downstream pairs, and basin topology
- **Risk**: Assess station-level data reliability, coverage gaps, and anomaly counts

### Advanced Deterministic Analysis
| Module | What it does |
|---|---|
| **Extreme Value** | Fit GEV / Gumbel distributions to annual maxima; estimate return levels (2–200 yr) with bootstrap confidence intervals |
| **Changepoint Detection** | Identify structural breaks in a time series using PELT; summarise per-segment mean, trend (Mann-Kendall), and variability |
| **Decomposition** | STL decomposition into Trend, Seasonal, and Residual components; quantifies seasonal and trend strength |
| **Wavelet Analysis** | Continuous Wavelet Transform (Morlet) with AR(1) red-noise significance contours and cone-of-influence boundary |

### Forecast & Project
| Module | What it does |
|---|---|
| **Prediction** | Multi-horizon forecasting using pre-trained FlowNet (TCN) or GraphNet (GNN) models with confidence bands |
| **Models** | Browse and compare available pre-trained models; inspect fit statistics and training metadata |
| **Scenario** | Simulate the downstream impact of a driver change (e.g. +20% rainfall for 3 months) on a target variable using lag-regression sensitivity analysis |
| **Climate** | Project long-term trends under SSP1-2.6, SSP2-4.5, and SSP5-8.5 emission scenarios using Sen's slope extrapolation |
| **Animate** | Step through historical observations as an animated map to visualise spatial patterns evolving over time |

---

## Feature Intelligence

HydroVision enforces a `FeatureRegistry` that prevents physically meaningless analysis combinations. For example, extreme value analysis is only offered for flow and level features (not temperature), and wavelet analysis is restricted to features with sufficient temporal resolution. This means the UI automatically adapts its options based on what the selected station and feature actually support.

---

## AI Narrative Insights

When a `GEMINI_API_KEY` is configured, the Analysis and Network modules use the Gemini LLM to generate plain-English summaries of statistical findings — interpreting trends, anomalies, and connectivity patterns in context. All AI-generated text is clearly labelled. If no key is provided (or the API is unavailable), the platform falls back to a deterministic rule-based summary so the app always works.

---

## Project Structure

```
hydrovision/
├── app.py                  # Flask entry point and route definitions
├── services/               # Backend logic — one service per analysis domain
│   ├── analysis_service.py
│   ├── chart_service.py
│   ├── changepoint_service.py
│   ├── climate_service.py
│   ├── decomposition_service.py
│   ├── extreme_service.py
│   ├── feature_registry.py
│   ├── network_service.py
│   ├── prediction_service.py
│   ├── quality_service.py
│   ├── scenario_service.py
│   └── wavelet_service.py
├── data/                   # Datasets (Mekong, LamaH-CE) — CSV files + index schemas
├── templates/              # Jinja2 HTML templates
├── static/
│   ├── js/app.js           # Single-page frontend controller (~6,000 lines)
│   └── css/                # Stylesheet modules
├── tests/                  # Pytest suite (2,769 tests)
└── docs/                   # Extended documentation
```

---

## Documentation

Full reference guides are in the `docs/` directory:

- [System Architecture](docs/architecture.md) — deterministic vs heuristic pipelines, data flow, service boundaries
- [Application API](docs/api.md) — all backend route signatures and response schemas
- [User Guide](docs/user-guide.md) — step-by-step workflows for each analysis module

---

## Local Development Setup

### Requirements
- Python 3.10+
- pip
- Virtual environment (recommended)

### Installation

```bash
# Clone and enter the repository
git clone <url>
cd hydrovision

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

```bash
cp .env.example .env
```

Open `.env` and optionally add your Gemini API key to enable AI narrative features:

```
GEMINI_API_KEY=your_key_here
```

The app runs fully without a key — AI summaries fall back to deterministic text.

### Running

```bash
python app.py
```

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Running Tests

```bash
pytest tests/
```

The test suite covers all service modules (2,769 tests). No external services or API keys are required to run tests.

---

## Datasets

| Dataset | Region | Stations | Features | Period |
|---|---|---|---|---|
| Mekong | Southeast Asia (China → Vietnam) | ~40 | Discharge, Water Level, Rainfall, Sediment | 1960s–2020s |
| LamaH-CE | Central Europe (Alps & Danube basin) | ~850 | Discharge, Rainfall, Temperature, Snow | 1981–2017 |

Both datasets are bundled with the repository. No external data download is required.

---

## Tech Stack

- **Backend**: Python / Flask
- **Data**: Pandas, NumPy, SciPy, statsmodels
- **Visualisation**: Plotly (server-rendered JSON, client-rendered via plotly.js)
- **ML Models**: PyTorch (TCN / GNN inference only — training is offline)
- **AI Narrative**: Google Gemini API (optional)
- **Frontend**: Vanilla JavaScript (no framework), CSS custom properties
- **Deployment**: Render (persistent disk for model weights)
