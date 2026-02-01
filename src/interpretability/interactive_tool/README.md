# Interactive Case Explorer

A Streamlit-based tool for manipulating process cases, running predictions, and comparing attributions.

## Features

- **Model Selection**: Choose from configured models (Helpdesk, DomesticDeclarations, etc.)
- **Case Browser**: Search and select cases from the test dataset
- **Clickable Event Editing**: Click on any event in the activity sequence to open an editing dialog
- **Full Feature Control**: Modify all categorical and numerical features in the popup editor
- **Side-by-Side Prediction Comparison**: Compare original vs modified case predictions
- **Attribution Analysis**: Compute and compare feature attributions between cases
- **Attribution Difference Heatmap**: Visualize how attributions change with modifications

## Running the App

From the project root directory:

```bash
pipenv run streamlit run src/interpretability/interactive_tool/app.py
```

Or if you're in the pipenv shell:

```bash
streamlit run src/interpretability/interactive_tool/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage Guide

1. **Select Model**: Choose a model from the sidebar dropdown
2. **Select Case**: Search or browse cases, then click "Load Case"
3. **Adjust Prefix Length**: Use the slider to set how many events to use as input
4. **Edit Events**: Click on any event button in the activity sequence to open the edit dialog
   - Modified events are highlighted in orange
   - Use "Reset to Original" to undo changes to a single event
   - Use "Reset All to Original" to undo all changes
5. **Enable Attributions**: Check "Compute Attributions" in the sidebar for attribution analysis
6. **Run Prediction**: Click the "Run Prediction" button
7. **Compare Results**: View side-by-side predictions and attribution heatmaps

## Attribution Methods

The tool uses **Input Perturbation (Leave-One-Out)** attribution:
- For each feature at each timestep, the feature is zeroed out
- The change in prediction probability indicates feature importance
- More interpretable than gradient-based methods
- Computationally more expensive but provides clear insights

## Architecture

```
interactive_tool/
├── app.py                 # Main Streamlit application
├── config.py              # Model configuration
├── model_manager.py       # Model and data loading
├── case_editor.py         # Case manipulation logic
├── prediction_engine.py   # Model inference
├── attribution_engine.py  # Attribution computation
├── components/            # UI components (for future expansion)
└── README.md
```

## Adding New Models

Edit `config.py` to add new models to the `AVAILABLE_MODELS` list:

```python
ModelConfig(
    name="my_model",                    # Internal identifier
    display_name="My Model",            # Display name in UI
    model_path="path/to/model.pkl",     # Relative to project root
    test_data_path="path/to/test.pkl",  # Relative to project root
    concept_name="Activity",            # Main activity feature name
    all_cat=["Activity", "Resource"],   # Categorical features for evaluation
    all_num=["case_elapsed_time"],      # Numerical features for evaluation
    growing_num_values=["case_elapsed_time"],  # Features that grow over time
)
```

## Visual Guide

### Activity Sequence Colors
- 🔵 **Blue**: Prefix events (used as model input)
- ⚪ **Gray**: Suffix events (ground truth, not used for prediction)
- 🟠 **Orange with ✏️**: Modified events

### Attribution Heatmaps
- **Red**: High positive attribution (feature strongly supports the prediction)
- **Blue**: Low/negative attribution (feature opposes the prediction)
- **White**: Neutral attribution
