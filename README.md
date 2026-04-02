# HydroVision

A modern, Flask-based spatial analytics dashboard built specifically for deep temporal exploration of hydrological monitoring stations.

## Overview
HydroVision has evolved into a highly scalable, multi-dataset hydrological engine. It is designed to combine absolute deterministic algorithms with advanced heuristic AI pipelines, supporting diverse geographical subsets ranging from the **Mekong Database** to the Central European **LamaH-CE** dataset.

## Features
- **Unified Multi-Dataset Architecture**: Dynamically load, map, and merge spatial data across fully autonomous datasets (Mekong and LamaH).
- **Intelligent Validations**: Employs a strict `FeatureRegistry` that automatically limits illogical feature selections based on real physics and dataset availability.
- **Robust Visualization Dock**: Build persistent comparative cards ranging from Flow Duration Curves to Multi-Station Temporal Heatmaps.
- **Advanced Automated Analysis**: 
  - **Deterministic**: Exact mathematical models, including STL Decomposition, Continuous Wavelet Transforms, and Complex Causal Networks.
  - **Heuristic**: Automated statistical narratives and optional LLM-driven anomaly contextualization.
- **Predictive Modelling**: Neural-network integration (e.g. Temporal Convolutional Networks/Graph Neural Networks) offering confidence bands and multi-horizon multi-variate forecasting where underlying pre-trained models are found in the deployment tree. 

## Project Structure
- `app.py` - Flask web-application entry point.
- `services/` - Microservice backend orchestrating data loading, pure chart JSON generation, deterministic math logic, and heuristic forecasting.
- `docs/` - Advanced reference documentation detailing API surfaces, architecture, and user workflows.
- `data/` - The physical datasets. Subdirectories like `Mekong` and `LamaH` house configurations, index schemas, and raw `.csv` historical files.
- `templates/` & `static/` - UI layouts, CSS abstractions, and Vanilla JavaScript frontend controllers.

## Documentation
Please refer to the following comprehensive guides in the `docs` directory:
- [System Architecture](docs/architecture.md) (Learn what is Truth vs Heurisitc prediction)
- [Application API](docs/api.md) (Backend server routes)
- [User Guide](docs/user-guide.md) (Standard operating procedure for the UI)

## Local Development Setup

### 1. Requirements
- Python 3.10+
- Pip
- Virtual Environment (recommended)

### 2. Installation
```bash
# Clone the repository and navigate inside
git clone <url>
cd hydrovision

# Initialize your virtual environment
python -m venv .venv
source .venv/bin/activate  # Or `.venv\Scripts\activate` on Windows

# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment Config
Copy the `.env.example` strictly to `.env`.
```bash
cp .env.example .env
```
*(Optionally provide your Gemini SDK key inside `.env` to unlock LLM-powered narrative insights in the Analysis Dock).*

### 4. Running
```bash
python app.py
```
Open `http://127.0.0.1:5000` to begin interacting.
