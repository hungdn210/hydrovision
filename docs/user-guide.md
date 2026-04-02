# User Guide

This guide describes how to operate the HydroVision web dashboard.

## Overview of Docks
The UI relies on an infinite multi-docking logic layer located on the right or bottom of the primary Leaflet map.

### 1. Visualization Dock
When clicking on a station on the interactive map, you pin parameters. By providing a date-range and feature array (e.g. `Discharge` and `Water Level`), clicking "Generate Visualization" sends the parameters to Python.
These visualizations are strictly trustworthy, plotting exact measurements (subject natively to `.dropna()` logic for missing chunks in physical datasets).

### 2. Analysis Dock
The analysis dock bridges visualization logic with heuristics.
- **Narrative Extraction**: Check the "Use AI" toggle if your `.env` contains an active Gemini key. This will generate dynamic NLP markdown summarizing statistical means instead of standard tabular facts.
- **Cross-Docking**: You can pass "Visualizations" *into* the Analysis API to execute operations specifically over the time bounds drawn inside the charts.

### 3. Prediction Dock
Run deep learning inferences into the future (or historically reconstruct missing segments).
> **Limitation Warning**: Generating Model Predictions strictly searches for valid pre-trained asset binaries located inside nested `data/` checkpoints. If the local server environment lacks the precise weights for a queried model strategy, the API gracefully falls back with a human-readable alert in the UI that the assets are currently un-installed for that method/station.

### 4. Advanced Logic: Scenarios
This feature permits you to override independent variable curves and watch dependent anomalies shift. An example would be artificially raising the `Rainfall` variables by 20% on the frontend and recalculating historical predicted `Discharge` limits to model theoretical flood consequences. 

*(Note: Invalid scenarios where the target feature is not logically aligned with the driver are safely blocked by UI capability enforcement and backend `scenario_service.py` exceptions).*
