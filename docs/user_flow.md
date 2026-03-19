# User Flow - Bridge SHM Dashboard

## Entry
1. User opens `streamlit` dashboard.
2. Dashboard checks for trained model/artifacts.
3. If artifacts are missing, pipeline auto-trains model from Kaggle dataset and prepares bridge data views.

## Bridge Selection
1. User clicks a bridge marker on the map.
2. App stores selected `bridge_id` in session state.
3. App auto-runs inference for the selected bridge (`run_bridge_inference`).

## Analytics Experience
1. App renders 3D bridge model.
2. Predicted anomalies are overlaid as 3D markers on deck/cables/piers.
3. User inspects anomaly intensity via marker color and size.

## Modality Inspection
1. GNSS trajectories are loaded and plotted.
2. InSAR LOS timeseries is loaded and plotted.
3. Sensor streams are loaded and plotted.
4. InSAR frame viewer shows raw image, mask prediction, and overlay.

## Operations
1. User can manually re-run processing for selected bridge.
2. App updates anomaly table and visuals in place.
