# Application API Reference

HydroVision leverages a Flask REST-like interface for asynchronous UI behavior.

*(Inputs and Outputs generally serialize to/from common strict JSON payloads)*

## Core Data APIs

### `GET /bootstrap`
**Description:** Essential startup payload syncing the exact current physical availability of stations and subsets. Let the frontend build navigation maps and filter dropdowns intelligently.
**Outputs:** 
- `stations`: List of JSON objects defining names and geo-coords.
- `graph_types`: Valid analysis operations.
- `feature_counts`: Frequency of occurrences logically categorized.

### `GET /datasets/<dataset>/data/<station>`
**Description:** Fetches a direct DataFrame abstraction bound closely to a physical `.csv` representation. 

## Analysis APIs

### `POST /analyse_free_multi_card`
**Description:** Used inside the Analysis Dock. Evaluates complex anomaly logic or summary statistics bounds on a filtered range. 
**Request JSON Context:**
```json
{
  "graphs": [  // array of previously serialized visual cards ],
  "useAI": true
}
```
**Returns:** Narrative Markdown block + Structured anomaly array findings.

### `POST /predict_multi_card`
**Description:** Routes configurations to the `prediction_service.py`. Automatically checks pre-trained weights for the active physical model matching the target features/geography. Fails fast if external assets are unavailable.

## Configuration Warnings
* **Feature Registry Strictness**: If the `/extreme_analyze` api is pushed a payload containing a `Temperature` request feature, it will throw a `ValueError` because the registry specifically forbids non-Flow / non-Precip datasets from being evaluated using extreme-value statistics.
