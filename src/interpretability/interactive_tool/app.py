"""
Interactive Case Explorer - Streamlit Application

A tool for manipulating process cases, running predictions, and comparing attributions.

Run with: streamlit run src/interpretability/interactive_tool/app.py
"""

import sys
from pathlib import Path

# Setup path to project root
_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / 'src').is_dir():
        break
    _current = _current.parent

PROJECT_ROOT = _current

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.interpretability.interactive_tool.config import (
    get_available_model_names,
    get_available_model_display_names,
    get_model_config
)
from src.interpretability.interactive_tool.model_manager import ModelManager
from src.interpretability.interactive_tool.case_editor import CaseEditor, EditableCase, Event
from src.interpretability.interactive_tool.prediction_engine import PredictionEngine
from src.interpretability.interactive_tool.attribution_engine import AttributionEngine


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Case Explorer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — compact layout
st.markdown("""
<style>
    /* Global font size reduction */
    html, body, [class*="css"] {
        font-size: 13px;
    }
    .main-header {
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .activity-chip {
        display: inline-block;
        padding: 4px 10px;
        margin: 2px;
        border-radius: 14px;
        font-size: 11px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .activity-chip:hover {
        transform: scale(1.05);
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
    .prefix-chip {
        background-color: #1f77b4;
        color: white;
    }
    .suffix-chip {
        background-color: #aaa;
        color: white;
    }
    .modified-chip {
        background-color: #ff7f0e;
        color: white;
        border: 2px dashed #cc6600;
    }
    /* Tighter block spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    /* Smaller headings */
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.0rem !important; }
    /* Compact sidebar */
    section[data-testid="stSidebar"] {
        font-size: 12px;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        font-size: 12px;
    }
    /* Smaller buttons */
    .stButton > button {
        font-size: 12px;
        padding: 0.25rem 0.75rem;
    }
    /* Compact expander */
    .streamlit-expanderHeader {
        font-size: 13px;
    }
    /* Tighter widget spacing */
    .stSelectbox, .stNumberInput, .stTextInput, .stSlider {
        margin-bottom: -0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialization
# ============================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'selected_model': None,
        'selected_case': None,
        'original_case': None,
        'modified_case': None,
        'prefix_length': 3,
        'prediction_result': None,
        'original_prediction': None,
        'cat_info': None,
        'num_info': None,
        'editing_event_idx': None,
        'original_attribution': None,
        'modified_attribution': None,
        'show_attributions': False,
        'case_level_cat': set(),
        'case_level_num': set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Helper Functions
# ============================================================

@st.cache_resource
def get_model_manager():
    """Get cached model manager."""
    import warnings
    warnings.filterwarnings('ignore')
    return ModelManager(PROJECT_ROOT)


def load_case_into_session(model_name: str, case_name: str):
    """Load a case into session state."""
    manager = get_model_manager()
    config = get_model_config(model_name)

    cat_info = manager.get_categorical_info(model_name)
    num_info = manager.get_numerical_info(model_name)

    st.session_state.cat_info = cat_info
    st.session_state.num_info = num_info

    cat_names = [c.name for c in cat_info]
    num_names = [n.name for n in num_info]

    editor = CaseEditor(cat_names, num_names)
    case_data = manager.get_case_data(model_name, case_name)
    editable = editor.tensor_to_editable(case_data, activity_feature=config.concept_name)

    st.session_state.original_case = editable
    st.session_state.modified_case = editable.copy()
    st.session_state.selected_case = case_name
    st.session_state.prefix_length = min(3, len(editable))
    st.session_state.prediction_result = None
    st.session_state.original_prediction = None
    st.session_state.original_attribution = None
    st.session_state.modified_attribution = None
    st.session_state.editing_event_idx = None

    # Detect case-level (constant) features
    case_level_cat, case_level_num = get_case_level_features(model_name, editable)
    st.session_state.case_level_cat = case_level_cat
    st.session_state.case_level_num = case_level_num


def get_activity_name(event: Event, activity_feature: str, cat_info) -> str:
    """Get human-readable activity name from event."""
    act_idx = event.categorical.get(activity_feature, 0)
    for cat in cat_info:
        if cat.name == activity_feature:
            return cat.idx_to_value.get(act_idx, f"Unknown_{act_idx}")
    return f"Unknown_{act_idx}"


def is_event_modified(original_event: Event, modified_event: Event) -> bool:
    """Check if an event has been modified."""
    for key in original_event.categorical:
        if original_event.categorical.get(key) != modified_event.categorical.get(key):
            return True
    for key in original_event.numerical:
        if abs(original_event.numerical.get(key, 0) - modified_event.numerical.get(key, 0)) > 1e-6:
            return True
    return False


def detect_constant_features(case: EditableCase) -> tuple:
    """Detect features that have the same value across all events in the case.

    Returns:
        (constant_cat, constant_num): Sets of feature names that are constant
    """
    if not case.events:
        return set(), set()

    constant_cat = set()
    constant_num = set()

    # Check categorical features
    for feat_name in case.cat_feature_names:
        values = [e.categorical.get(feat_name) for e in case.events]
        if len(set(values)) == 1:
            constant_cat.add(feat_name)

    # Check numerical features
    for feat_name in case.num_feature_names:
        values = [e.numerical.get(feat_name, 0.0) for e in case.events]
        # Check if all values are approximately equal
        if values and max(values) - min(values) < 1e-6:
            constant_num.add(feat_name)

    return constant_cat, constant_num


def get_case_level_features(model_name: str, case: EditableCase) -> tuple:
    """Get case-level (constant) features, either from config or auto-detected.

    Returns:
        (case_level_cat, case_level_num): Sets of feature names
    """
    config = get_model_config(model_name)

    # Use config if specified, otherwise auto-detect
    if config.case_level_cat is not None:
        case_level_cat = set(config.case_level_cat)
    else:
        case_level_cat, _ = detect_constant_features(case)

    if config.case_level_num is not None:
        case_level_num = set(config.case_level_num)
    else:
        _, case_level_num = detect_constant_features(case)

    return case_level_cat, case_level_num


# ============================================================
# Event Editor Dialog
# ============================================================

@st.dialog("Edit Event", width="large")
def edit_event_dialog(event_idx: int):
    """Dialog for editing a single event (excludes case-level features)."""
    if st.session_state.modified_case is None:
        st.error("No case loaded")
        return

    event = st.session_state.modified_case.events[event_idx]
    original_event = st.session_state.original_case.events[event_idx]
    cat_info = st.session_state.cat_info
    num_info = st.session_state.num_info
    config = get_model_config(st.session_state.selected_model)

    # Get case-level features (to exclude from event editing)
    case_level_cat = st.session_state.case_level_cat
    case_level_num = st.session_state.case_level_num

    st.markdown(f"### Editing Event {event_idx + 1}")

    # Show original vs current
    activity_name = get_activity_name(event, config.concept_name, cat_info)
    original_activity = get_activity_name(original_event, config.concept_name, cat_info)

    if activity_name != original_activity:
        st.info(f"Original activity: **{original_activity}**")

    # Filter out case-level features
    event_cat_info = [c for c in cat_info if c.name not in case_level_cat]
    event_num_info = [n for n in num_info if n.name not in case_level_num]

    # Create tabs for categorical and numerical features
    tab_cat, tab_num = st.tabs(["Categorical Features", "Numerical Features"])

    with tab_cat:
        if not event_cat_info:
            st.info("All categorical features are case-level. Edit them in the Case Attributes section.")
        for cat in event_cat_info:
            current_val = event.categorical.get(cat.name, 0)
            original_val = original_event.categorical.get(cat.name, 0)

            options = list(range(cat.num_values))
            option_labels = {
                idx: f"{str(cat.idx_to_value.get(idx, 'Unknown'))[:40]}"
                for idx in options
            }

            # Highlight if different from original
            label = cat.name
            if current_val != original_val:
                label = f"🔸 {cat.name} (modified)"

            new_val = st.selectbox(
                label,
                options=options,
                index=current_val if current_val in options else 0,
                format_func=lambda x, ol=option_labels: ol.get(x, str(x)),
                key=f"dialog_cat_{event_idx}_{cat.name}"
            )

            if new_val != current_val:
                event.categorical[cat.name] = new_val

    with tab_num:
        if not event_num_info:
            st.info("All numerical features are case-level. Edit them in the Case Attributes section.")
        for num in event_num_info:
            current_val = event.numerical.get(num.name, 0.0)
            original_val = original_event.numerical.get(num.name, 0.0)

            label = num.name
            if abs(current_val - original_val) > 1e-6:
                label = f"🔸 {num.name} (modified)"

            new_val = st.number_input(
                label,
                value=float(current_val),
                format="%.4f",
                key=f"dialog_num_{event_idx}_{num.name}"
            )

            if abs(new_val - current_val) > 1e-6:
                event.numerical[num.name] = new_val

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Reset to Original", use_container_width=True):
            st.session_state.modified_case.events[event_idx] = original_event.copy()
            st.rerun()

    with col2:
        if st.button("Apply Changes", type="primary", use_container_width=True):
            st.rerun()

    with col3:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ============================================================
# Case-Level Feature Editor
# ============================================================

def render_case_level_editor():
    """Render editor for case-level (constant) features."""
    case_level_cat = st.session_state.case_level_cat
    case_level_num = st.session_state.case_level_num

    if not case_level_cat and not case_level_num:
        return  # No case-level features to show

    cat_info = st.session_state.cat_info
    num_info = st.session_state.num_info
    modified_case = st.session_state.modified_case
    original_case = st.session_state.original_case

    # Filter to only case-level features
    case_cat_info = [c for c in cat_info if c.name in case_level_cat]
    case_num_info = [n for n in num_info if n.name in case_level_num]

    n_case_features = len(case_cat_info) + len(case_num_info)
    with st.expander(f"📋 Case Attributes ({n_case_features} constant features)", expanded=True):
        st.caption("These features have the same value for all events in the case. Changes apply to all events.")

        changed = False

        # Categorical features
        if case_cat_info:
            st.markdown("**Categorical**")
            # Create rows of 3 columns each
            for row_start in range(0, len(case_cat_info), 3):
                row_items = case_cat_info[row_start:row_start + 3]
                cols = st.columns(len(row_items))
                for col_idx, cat in enumerate(row_items):
                    with cols[col_idx]:
                        # Get current value from first event
                        current_val = modified_case.events[0].categorical.get(cat.name, 0)
                        original_val = original_case.events[0].categorical.get(cat.name, 0)

                        options = list(range(cat.num_values))
                        option_labels = {
                            idx: f"{str(cat.idx_to_value.get(idx, 'Unknown'))[:30]}"
                            for idx in options
                        }

                        label = cat.name
                        if current_val != original_val:
                            label = f"🔸 {cat.name}"

                        new_val = st.selectbox(
                            label,
                            options=options,
                            index=current_val if current_val in options else 0,
                            format_func=lambda x, ol=option_labels: ol.get(x, str(x)),
                            key=f"case_level_cat_{cat.name}"
                        )

                        if new_val != current_val:
                            # Apply to ALL events
                            for event in modified_case.events:
                                event.categorical[cat.name] = new_val
                            changed = True

        # Numerical features
        if case_num_info:
            st.markdown("**Numerical**")
            for row_start in range(0, len(case_num_info), 3):
                row_items = case_num_info[row_start:row_start + 3]
                cols = st.columns(len(row_items))
                for col_idx, num in enumerate(row_items):
                    with cols[col_idx]:
                        current_val = modified_case.events[0].numerical.get(num.name, 0.0)
                        original_val = original_case.events[0].numerical.get(num.name, 0.0)

                        label = num.name
                        if abs(current_val - original_val) > 1e-6:
                            label = f"🔸 {num.name}"

                        new_val = st.number_input(
                            label,
                            value=float(current_val),
                            format="%.4f",
                            key=f"case_level_num_{num.name}"
                        )

                        if abs(new_val - current_val) > 1e-6:
                            # Apply to ALL events
                            for event in modified_case.events:
                                event.numerical[num.name] = new_val
                            changed = True

        if changed:
            st.rerun()


# ============================================================
# Activity Sequence Display
# ============================================================

def render_clickable_sequence(case: EditableCase, original_case: EditableCase, prefix_length: int, cat_info, activity_feature: str):
    """Render clickable activity sequence."""
    st.markdown("### Activity Sequence")
    st.caption("Click on any event to edit it")

    # Create columns for events
    events_per_row = 6
    total_events = len(case.events)

    for row_start in range(0, total_events, events_per_row):
        cols = st.columns(min(events_per_row, total_events - row_start))

        for i, col in enumerate(cols):
            event_idx = row_start + i
            if event_idx >= total_events:
                break

            event = case.events[event_idx]
            original_event = original_case.events[event_idx]
            activity_name = get_activity_name(event, activity_feature, cat_info)
            is_modified = is_event_modified(original_event, event)
            is_prefix = event_idx < prefix_length

            # Determine styling
            if is_modified:
                bg_color = "#ff7f0e"
                border = "2px dashed #cc6600"
            elif is_prefix:
                bg_color = "#1f77b4"
                border = "none"
            else:
                bg_color = "#aaa"
                border = "none"

            with col:
                # Truncate long names
                display_name = activity_name[:15] + "..." if len(activity_name) > 15 else activity_name
                button_label = f"{event_idx + 1}. {display_name}"

                if is_modified:
                    button_label = f"✏️ {button_label}"

                if st.button(
                    button_label,
                    key=f"event_btn_{event_idx}",
                    use_container_width=True,
                    type="secondary" if not is_prefix else "primary"
                ):
                    edit_event_dialog(event_idx)

    # Legend
    st.markdown("---")
    legend_cols = st.columns(4)
    with legend_cols[0]:
        st.markdown("🔵 **Prefix events** (used for prediction)")
    with legend_cols[1]:
        st.markdown("⚪ **Suffix events** (ground truth)")
    with legend_cols[2]:
        st.markdown("🟠 **Modified events**")
    with legend_cols[3]:
        if st.button("🔄 Reset All to Original"):
            st.session_state.modified_case = st.session_state.original_case.copy()
            st.session_state.prediction_result = None
            st.session_state.original_attribution = None
            st.session_state.modified_attribution = None
            st.rerun()


# ============================================================
# Attribution Visualization
# ============================================================

def create_attribution_heatmap(attribution_result, title: str):
    """Create a heatmap visualization of attributions."""
    if attribution_result is None:
        return None

    data, event_labels, feature_names = attribution_result.to_heatmap_data()

    if data.size == 0:
        return None

    # Transpose for better visualization (features as rows, events as columns)
    data = data.T

    fig = px.imshow(
        data,
        labels=dict(x="Event", y="Feature", color="Attribution"),
        x=event_labels,
        y=feature_names,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=max(300, len(feature_names) * 25),
        margin=dict(l=150)
    )

    return fig


def render_attribution_comparison():
    """Render side-by-side attribution comparison."""
    if not st.session_state.show_attributions:
        return

    st.markdown("---")
    st.markdown("## 🎯 Attribution Analysis")

    orig_attr = st.session_state.original_attribution
    mod_attr = st.session_state.modified_attribution

    if orig_attr is None and mod_attr is None:
        st.info("Run prediction with 'Compute Attributions' enabled to see attribution analysis")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Case Attribution")
        if orig_attr:
            st.caption(f"Target: {orig_attr.target_activity}")
            fig = create_attribution_heatmap(orig_attr, "Original Case")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="orig_attr_heatmap")
        else:
            st.info("No attribution computed")

    with col2:
        st.markdown("### Modified Case Attribution")
        if mod_attr:
            st.caption(f"Target: {mod_attr.target_activity}")
            fig = create_attribution_heatmap(mod_attr, "Modified Case")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="mod_attr_heatmap")
        else:
            st.info("No attribution computed")

    # Show attribution difference if both are available
    if orig_attr and mod_attr and len(orig_attr.event_attributions) == len(mod_attr.event_attributions):
        st.markdown("### Attribution Difference (Modified - Original)")

        diff_events = []
        for orig_e, mod_e in zip(orig_attr.event_attributions, mod_attr.event_attributions):
            diff_dict = {k: mod_e.get(k, 0) - orig_e.get(k, 0) for k in orig_e.keys()}
            diff_events.append(diff_dict)

        # Create difference heatmap
        if diff_events:
            features = list(diff_events[0].keys())
            data = np.array([[e.get(f, 0) for f in features] for e in diff_events]).T
            event_labels = [f"Event {i+1}" for i in range(len(diff_events))]

            fig = px.imshow(
                data,
                labels=dict(x="Event", y="Feature", color="Difference"),
                x=event_labels,
                y=features,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto"
            )
            fig.update_layout(
                title=dict(text="Attribution Change", x=0.5),
                height=max(300, len(features) * 25),
                margin=dict(l=150)
            )
            st.plotly_chart(fig, use_container_width=True, key="attr_diff_heatmap")


# ============================================================
# Prediction Results
# ============================================================

def create_probability_chart(prediction_result, step=0):
    """Create probability distribution chart for a prediction step."""
    if not prediction_result or step >= len(prediction_result.top_k_predictions):
        return None

    top_k = prediction_result.top_k_predictions[step]
    activities = [t[0][:25] for t in top_k]
    probs = [t[1] * 100 for t in top_k]

    fig = px.bar(
        x=probs,
        y=activities,
        orientation='h',
        labels={'x': 'Probability (%)', 'y': 'Activity'},
        color=probs,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=200,
        showlegend=False,
        yaxis=dict(autorange='reversed'),
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    return fig


def render_prediction_results():
    """Render prediction results."""
    if not st.session_state.prediction_result:
        return

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    orig_pred = st.session_state.original_prediction
    mod_pred = st.session_state.prediction_result

    # Side by side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🟢 Original Case")
        if orig_pred:
            st.markdown(f"**Predicted:** {orig_pred.predicted_sequence_str}")
            if orig_pred.actual_suffix_activities:
                st.markdown(f"**Actual:** {' → '.join(orig_pred.actual_suffix_activities)}")

            st.markdown("**Top predictions (Step 1):**")
            prob_fig = create_probability_chart(orig_pred, 0)
            if prob_fig:
                st.plotly_chart(prob_fig, use_container_width=True, key="orig_prob_chart")

    with col2:
        st.markdown("### 🟠 Modified Case")
        if mod_pred:
            st.markdown(f"**Predicted:** {mod_pred.predicted_sequence_str}")

            st.markdown("**Top predictions (Step 1):**")
            prob_fig = create_probability_chart(mod_pred, 0)
            if prob_fig:
                st.plotly_chart(prob_fig, use_container_width=True, key="mod_prob_chart")

    # Check if predictions differ
    if orig_pred and mod_pred:
        if orig_pred.predicted_activities != mod_pred.predicted_activities:
            st.success("The modification changed the prediction!")
        else:
            st.info("The prediction remained the same despite the modification.")

    # Render attribution comparison
    render_attribution_comparison()


# ============================================================
# Sidebar
# ============================================================

def render_sidebar():
    """Render the sidebar."""
    st.sidebar.markdown("## 🎯 Model & Case Selection")

    # Model selection
    model_names = get_available_model_names()
    display_names = get_available_model_display_names()

    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=model_names,
        format_func=lambda x: display_names.get(x, x),
        key="model_selector"
    )

    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.selected_case = None
        st.session_state.original_case = None
        st.session_state.modified_case = None
        st.session_state.prediction_result = None
        st.session_state.original_attribution = None
        st.session_state.modified_attribution = None

    if selected_model:
        manager = get_model_manager()

        # Case selection
        st.sidebar.markdown("---")
        try:
            case_names = manager.get_case_names(selected_model)

            search_term = st.sidebar.text_input("🔍 Search cases", "")
            if search_term:
                filtered_cases = [c for c in case_names if search_term.lower() in c.lower()]
            else:
                filtered_cases = case_names[:100]

            selected_case = st.sidebar.selectbox(
                f"Select Case ({len(case_names)} total)",
                options=filtered_cases,
                key="case_selector"
            )

            if selected_case and st.sidebar.button("📥 Load Case", use_container_width=True):
                with st.spinner("Loading case..."):
                    load_case_into_session(selected_model, selected_case)
                st.rerun()

        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    # Prefix length slider
    if st.session_state.original_case:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Prediction Settings")

        max_len = len(st.session_state.original_case)
        st.session_state.prefix_length = st.sidebar.slider(
            "Prefix Length",
            min_value=1,
            max_value=max_len - 1 if max_len > 1 else 1,
            value=min(st.session_state.prefix_length, max_len - 1),
            help="Number of events to use as input"
        )

        # Attribution toggle
        st.session_state.show_attributions = st.sidebar.checkbox(
            "Compute Attributions",
            value=st.session_state.show_attributions,
            help="Compute and compare attributions (slower)"
        )

        # Run prediction button
        st.sidebar.markdown("---")
        if st.sidebar.button("🔮 Run Prediction", type="primary", use_container_width=True):
            run_prediction()


def run_prediction():
    """Run prediction on both original and modified cases."""
    if not st.session_state.modified_case or not st.session_state.selected_model:
        return

    manager = get_model_manager()
    model = manager.load_model(st.session_state.selected_model)
    config = get_model_config(st.session_state.selected_model)

    cat_info = st.session_state.cat_info
    num_info = st.session_state.num_info

    cat_names = [c.name for c in cat_info]
    num_names = [n.name for n in num_info]

    activity_to_idx, idx_to_activity = manager.get_activity_mapping(st.session_state.selected_model)

    # Create prediction engine
    engine = PredictionEngine(
        model=model,
        cat_feature_names=cat_names,
        num_feature_names=num_names,
        activity_feature=config.concept_name,
        idx_to_activity=idx_to_activity,
        activity_to_idx=activity_to_idx
    )

    # Predict on original
    st.session_state.original_prediction = engine.predict_from_editable_case(
        st.session_state.original_case,
        st.session_state.prefix_length,
        max_suffix_length=15,
        top_k=5
    )

    # Predict on modified
    st.session_state.prediction_result = engine.predict_from_editable_case(
        st.session_state.modified_case,
        st.session_state.prefix_length,
        max_suffix_length=15,
        top_k=5
    )

    # Compute attributions if enabled
    if st.session_state.show_attributions:
        attr_engine = AttributionEngine(
            model=model,
            cat_feature_names=cat_names,
            num_feature_names=num_names,
            activity_feature=config.concept_name,
            idx_to_activity=idx_to_activity
        )

        # Get prefix tensors for attribution
        case_editor = CaseEditor(cat_names, num_names)

        # Original case attributions
        orig_prefix, _ = case_editor.create_prefix_suffix_split(
            st.session_state.original_case,
            st.session_state.prefix_length
        )
        orig_target_idx = st.session_state.original_prediction.predicted_activity_indices[0] if st.session_state.original_prediction.predicted_activity_indices else None

        st.session_state.original_attribution = attr_engine.compute_input_perturbation_attribution(
            orig_prefix[0], orig_prefix[1],
            target_activity_idx=orig_target_idx
        )

        # Modified case attributions
        mod_prefix, _ = case_editor.create_prefix_suffix_split(
            st.session_state.modified_case,
            st.session_state.prefix_length
        )
        mod_target_idx = st.session_state.prediction_result.predicted_activity_indices[0] if st.session_state.prediction_result.predicted_activity_indices else None

        st.session_state.modified_attribution = attr_engine.compute_input_perturbation_attribution(
            mod_prefix[0], mod_prefix[1],
            target_activity_idx=mod_target_idx
        )


# ============================================================
# Main
# ============================================================

def main():
    """Main application."""
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🔮 Interactive Case Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Click on events to modify them, then compare predictions and attributions</p>', unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Main content
    if not st.session_state.selected_model:
        st.info("👈 Select a model from the sidebar to get started")
        return

    if not st.session_state.original_case:
        st.info("👈 Select and load a case from the sidebar")
        return

    # Get model config
    config = get_model_config(st.session_state.selected_model)
    cat_info = st.session_state.cat_info

    # Render case-level feature editor (constant features across all events)
    render_case_level_editor()

    # Render clickable sequence
    render_clickable_sequence(
        st.session_state.modified_case,
        st.session_state.original_case,
        st.session_state.prefix_length,
        cat_info,
        config.concept_name
    )

    # Render prediction results
    render_prediction_results()


if __name__ == "__main__":
    main()
