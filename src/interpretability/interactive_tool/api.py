"""
FastAPI application for the Interactive Case Explorer.

Run with:
    uvicorn src.interpretability.interactive_tool.api:app --reload --port 8000
"""

import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ---------------------------------------------------------------------------
# Project root discovery (same logic as app.py)
# ---------------------------------------------------------------------------
_current = Path(__file__).resolve().parent
while _current != _current.parent:
    if (_current / "src").is_dir():
        break
    _current = _current.parent
PROJECT_ROOT = _current

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.interpretability.interactive_tool.config import (
    get_available_model_display_names,
    get_available_model_names,
    get_model_config,
)
from src.interpretability.interactive_tool.model_manager import ModelManager
from src.interpretability.interactive_tool.case_editor import (
    CaseEditor,
    EditableCase,
    Event,
    get_case_level_features,
)
from src.interpretability.interactive_tool.prediction_engine import PredictionEngine
from src.interpretability.interactive_tool.api_attribution import compute_ig_attribution
from src.interpretability.interactive_tool.api_schemas import (
    AttributionRequest,
    AttributionResponse,
    CaseLoadResponse,
    CasesResponse,
    CategoricalFeatureInfo,
    EditableCaseSchema,
    EventSchema,
    ModelInfo,
    ModelsResponse,
    NumericalFeatureInfo,
    PredictRequest,
    PredictResponse,
    PredictionStepResult,
    TopKPrediction,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

app = FastAPI(title="Interactive Case Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton ModelManager – created once at import time
_manager = ModelManager(PROJECT_ROOT)

# PredictionEngine cache: model_name -> PredictionEngine
_prediction_engines: Dict[str, PredictionEngine] = {}

# ProcessPerformanceMap cache: model_name -> PPM (lazy, expensive first call)
_ppm_cache: Dict[str, "ProcessPerformanceMap"] = {}


def _get_prediction_engine(model_name: str) -> PredictionEngine:
    if model_name not in _prediction_engines:
        config = get_model_config(model_name)
        model = _manager.load_model(model_name)
        cat_info = _manager.get_categorical_info(model_name)
        num_info = _manager.get_numerical_info(model_name)
        cat_names = [c.name for c in cat_info]
        num_names = [n.name for n in num_info]
        activity_to_idx, idx_to_activity = _manager.get_activity_mapping(model_name)

        window_size = _manager.get_window_size(model_name)

        engine = PredictionEngine(
            model=model,
            cat_feature_names=cat_names,
            num_feature_names=num_names,
            activity_feature=config.concept_name,
            idx_to_activity=idx_to_activity,
            activity_to_idx=activity_to_idx,
            window_size=window_size,
        )
        _prediction_engines[model_name] = engine
    return _prediction_engines[model_name]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _schema_to_editable(schema: EditableCaseSchema) -> EditableCase:
    events = [
        Event(categorical=dict(e.categorical), numerical=dict(e.numerical))
        for e in schema.events
    ]
    return EditableCase(
        events=events,
        cat_feature_names=list(schema.cat_feature_names),
        num_feature_names=list(schema.num_feature_names),
        original_length=schema.original_length,
    )


def _editable_to_schema(case: EditableCase) -> EditableCaseSchema:
    events = [
        EventSchema(categorical=dict(e.categorical), numerical=dict(e.numerical))
        for e in case.events
    ]
    return EditableCaseSchema(
        events=events,
        cat_feature_names=case.cat_feature_names,
        num_feature_names=case.num_feature_names,
        original_length=case.original_length,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/models", response_model=ModelsResponse)
async def list_models():
    """List available models with display names."""
    names = get_available_model_names()
    display = get_available_model_display_names()
    return ModelsResponse(
        models=[ModelInfo(name=n, display_name=display[n]) for n in names]
    )


@app.get("/api/models/{name}/cases", response_model=CasesResponse)
async def list_cases(
    name: str,
    search: str = Query("", description="Filter case names"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    """Paginated / searchable case names."""
    if name not in get_available_model_names():
        raise HTTPException(404, f"Model '{name}' not found")

    all_cases = await run_in_threadpool(_manager.get_case_names, name)
    if search:
        filtered = [c for c in all_cases if search.lower() in c.lower()]
    else:
        filtered = all_cases

    total = len(filtered)
    page = filtered[offset: offset + limit]
    return CasesResponse(cases=page, total=total, offset=offset, limit=limit)


@app.get("/api/models/{name}/cases/{case_name}", response_model=CaseLoadResponse)
async def load_case(name: str, case_name: str):
    """Load a case and return its editable representation + metadata."""
    if name not in get_available_model_names():
        raise HTTPException(404, f"Model '{name}' not found")

    def _load():
        config = get_model_config(name)
        cat_info = _manager.get_categorical_info(name)
        num_info = _manager.get_numerical_info(name)
        cat_names = [c.name for c in cat_info]
        num_names = [n.name for n in num_info]

        editor = CaseEditor(cat_names, num_names)
        case_data = _manager.get_case_data(name, case_name)
        editable = editor.tensor_to_editable(case_data, activity_feature=config.concept_name)

        cl_cat, cl_num = get_case_level_features(name, editable)

        return editable, cat_info, num_info, config.concept_name, cl_cat, cl_num

    try:
        editable, cat_info, num_info, concept_name, cl_cat, cl_num = await run_in_threadpool(_load)
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    def _sanitize_str(v) -> str:
        """Convert nan / non-string values to a safe string."""
        if isinstance(v, float) and math.isnan(v):
            return "<NA>"
        return str(v)

    return CaseLoadResponse(
        case=_editable_to_schema(editable),
        cat_info=[
            CategoricalFeatureInfo(
                name=c.name,
                num_values=c.num_values,
                idx_to_value={int(k): _sanitize_str(v) for k, v in c.idx_to_value.items()},
                value_to_idx={_sanitize_str(k): v for k, v in c.value_to_idx.items()},
            )
            for c in cat_info
        ],
        num_info=[NumericalFeatureInfo(name=n.name) for n in num_info],
        concept_name=concept_name,
        case_level_cat=sorted(cl_cat),
        case_level_num=sorted(cl_num),
    )


@app.post("/api/models/{name}/predict", response_model=PredictResponse)
async def predict(name: str, body: PredictRequest):
    """Run autoregressive prediction on an EditableCase."""
    if name not in get_available_model_names():
        raise HTTPException(404, f"Model '{name}' not found")

    editable = _schema_to_editable(body.case)
    engine = _get_prediction_engine(name)

    result = await run_in_threadpool(
        engine.predict_from_editable_case,
        editable,
        body.prefix_length,
        body.max_suffix_length,
        body.top_k,
    )

    steps = []
    for i, top_k_list in enumerate(result.top_k_predictions):
        steps.append(PredictionStepResult(
            predicted_activity=result.predicted_activities[i],
            top_k=[TopKPrediction(activity=a, probability=p) for a, p in top_k_list],
        ))

    return PredictResponse(
        predicted_activities=result.predicted_activities,
        predicted_sequence=result.predicted_sequence_str,
        steps=steps,
        actual_suffix_activities=result.actual_suffix_activities,
    )


@app.post("/api/models/{name}/attribution", response_model=AttributionResponse)
async def attribution(name: str, body: AttributionRequest):
    """Compute IG attribution via InterpretabilityTool."""
    if name not in get_available_model_names():
        raise HTTPException(404, f"Model '{name}' not found")

    editable = _schema_to_editable(body.case)

    result = await run_in_threadpool(
        compute_ig_attribution,
        name,
        _manager,
        editable,
        body.prefix_length,
        body.target,
        body.target_class,
        body.suffix_step,
        body.n_steps,
        body.method,
    )
    return result


# ---------------------------------------------------------------------------
# Debug: compare PredictionEngine vs Evaluation pipeline
# ---------------------------------------------------------------------------

@app.get("/api/models/{name}/debug-predict")
async def debug_predict(
    name: str,
    case_name: str = Query(...),
    prefix_length: int = Query(..., ge=1),
):
    """Compare PredictionEngine output vs Evaluation._predict_suffix_with_means."""
    if name not in get_available_model_names():
        raise HTTPException(404, f"Model '{name}' not found")

    def _run():
        import torch

        config = get_model_config(name)
        engine = _get_prediction_engine(name)
        eval_helper = _manager.get_eval_helper(name)

        # --- Approach A: Evaluation pipeline (known-good) --------
        case_raw = eval_helper.cases[case_name]
        target_prefix = None
        target_prefix_len = None
        for pl, prefix, suffix in eval_helper._iterate_case(case_raw):
            if pl == prefix_length:
                # Deep-copy so the rolling iterator doesn't mutate it
                target_prefix = (
                    [t.clone() for t in prefix[0]],
                    [t.clone() for t in prefix[1]],
                )
                target_prefix_len = pl
                break

        eval_first_prediction = None
        eval_first_logits = None
        if target_prefix is not None:
            # Run model.inference directly (same as _predict_suffix_with_means)
            eval_helper._disable_model_dropout(eval_helper.model)
            prediction_a, _, _ = eval_helper.model.inference(prefix=target_prefix)
            eval_helper._enable_dropout(eval_helper.model, eval_helper._disable_model_dropout(eval_helper.model))

            act_key = config.concept_name + "_mean"
            logits_a = prediction_a[0][0][act_key].squeeze()
            probs_a = torch.nn.functional.softmax(logits_a, dim=-1)
            idx_a = probs_a.argmax().item()

            _, idx_to_act = _manager.get_activity_mapping(name)
            eval_first_prediction = idx_to_act.get(idx_a, f"Unk_{idx_a}")
            eval_first_logits = {
                idx_to_act.get(i, f"Unk_{i}"): round(probs_a[i].item(), 4)
                for i in range(len(probs_a))
            }

            # Also get prefix tensor info
            act_feat_idx = [c.name for c in _manager.get_categorical_info(name)].index(config.concept_name)
            eval_prefix_activities = target_prefix[0][act_feat_idx][0].tolist()
        else:
            eval_prefix_activities = []

        # --- Approach B: PredictionEngine --------------------------
        cat_info = _manager.get_categorical_info(name)
        num_info = _manager.get_numerical_info(name)
        cat_names = [c.name for c in cat_info]
        num_names = [n.name for n in num_info]
        from .case_editor import CaseEditor
        editor = CaseEditor(cat_names, num_names)
        case_data = _manager.get_case_data(name, case_name)
        editable = editor.tensor_to_editable(case_data, activity_feature=config.concept_name)

        prefix_tensors, suffix_tensors = editor.create_prefix_suffix_split(editable, prefix_length)

        # Build padded prefix (same as PredictionEngine._left_pad_tensors)
        ws = engine.window_size
        padded_cat = []
        for t in prefix_tensors[0]:
            p = torch.zeros(1, ws, dtype=torch.long)
            p[0, -t.shape[0]:] = t
            padded_cat.append(p)
        padded_num = []
        for t in prefix_tensors[1]:
            p = torch.zeros(1, ws, dtype=torch.float32)
            p[0, -t.shape[0]:] = t
            padded_num.append(p)

        padded_prefix_b = [padded_cat, padded_num]

        engine.model.eval()
        with torch.no_grad():
            prediction_b, _, _ = engine.model.inference(prefix=padded_prefix_b)

        act_key = config.concept_name + "_mean"
        logits_b = prediction_b[0][0][act_key].squeeze()
        probs_b = torch.nn.functional.softmax(logits_b, dim=-1)
        idx_b = probs_b.argmax().item()
        _, idx_to_act = _manager.get_activity_mapping(name)
        engine_first_prediction = idx_to_act.get(idx_b, f"Unk_{idx_b}")
        engine_first_logits = {
            idx_to_act.get(i, f"Unk_{i}"): round(probs_b[i].item(), 4)
            for i in range(len(probs_b))
        }

        engine_prefix_activities = padded_prefix_b[0][act_feat_idx][0].tolist()

        # Actual suffix (from editable case)
        actual_suffix_acts = [
            idx_to_act.get(ev.categorical.get(config.concept_name, 0), "?")
            for ev in editable.events[prefix_length:]
        ]

        # Tensor comparison
        tensors_match = True
        if target_prefix is not None:
            for j in range(len(target_prefix[0])):
                if not torch.equal(target_prefix[0][j], padded_prefix_b[0][j]):
                    tensors_match = False
                    break
            if tensors_match:
                for j in range(len(target_prefix[1])):
                    if not torch.equal(target_prefix[1][j], padded_prefix_b[1][j]):
                        tensors_match = False
                        break

        return {
            "case_name": case_name,
            "prefix_length": prefix_length,
            "window_size": ws,
            "num_events_in_editable_case": len(editable.events),
            "actual_suffix": actual_suffix_acts,
            "tensors_match": tensors_match,
            "eval_prefix_activities": eval_prefix_activities,
            "engine_prefix_activities": engine_prefix_activities,
            "eval_first_prediction": eval_first_prediction,
            "engine_first_prediction": engine_first_prediction,
            "eval_top5": dict(sorted(eval_first_logits.items(), key=lambda x: -x[1])[:5]) if eval_first_logits else None,
            "engine_top5": dict(sorted(engine_first_logits.items(), key=lambda x: -x[1])[:5]),
        }

    try:
        result = await run_in_threadpool(_run)
    except Exception as exc:
        import traceback
        raise HTTPException(500, f"{exc}\n{traceback.format_exc()}")
    return result


# ---------------------------------------------------------------------------
# Process Performance Map
# ---------------------------------------------------------------------------

def _get_ppm(model_name: str):
    """Get or create a cached ProcessPerformanceMap (expensive first call)."""
    if model_name not in _ppm_cache:
        from src.interpretability.visualization.process_performance_map import (
            ProcessPerformanceMap,
        )

        config = get_model_config(model_name)
        model = _manager.load_model(model_name)
        dataset = _manager.load_test_dataset(model_name)
        eval_helper = _manager.get_eval_helper(model_name)

        ppm = ProcessPerformanceMap(
            model=model,
            test_dataset=dataset,
            eval_helper=eval_helper,
            activity_key=config.concept_name,
        )
        ppm.collect_predictions(show_progress=True)
        _ppm_cache[model_name] = ppm

    return _ppm_cache[model_name]


def _build_highlighted_svg(
    ppm,
    highlight_activities: List[str],
    prefix_length: int,
) -> str:
    """Build process map SVG with the given case path highlighted.

    Edges and nodes on the case path get thick blue borders; prefix
    transitions are distinguished from suffix transitions.
    """
    import graphviz

    if ppm.dfg is None:
        ppm.mine_dfg()

    activity_name_map = ppm.get_activity_name_map()

    def display(aid: str) -> str:
        return activity_name_map.get(aid, aid)

    # Derive transition path from the activity list
    path_transitions: Set[Tuple[str, str]] = set()
    for i in range(len(highlight_activities) - 1):
        path_transitions.add((highlight_activities[i], highlight_activities[i + 1]))

    path_nodes: Set[str] = set(highlight_activities)

    prefix_transitions: Set[Tuple[str, str]] = set()
    for i in range(min(prefix_length - 1, len(highlight_activities) - 1)):
        prefix_transitions.add((highlight_activities[i], highlight_activities[i + 1]))

    prefix_nodes: Set[str] = set(highlight_activities[:prefix_length])

    # --- build graph ---
    dot = graphviz.Digraph(comment="Process Performance Map")
    dot.attr(rankdir="LR", bgcolor="white")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="white")

    all_activities: Set[str] = set()
    for src, tgt in ppm.dfg:
        all_activities.add(src)
        all_activities.add(tgt)

    for activity in all_activities:
        # node accuracy colour (same logic as PPM)
        total_correct = 0
        total_count = 0
        for (src, _), perf in ppm.transition_performance.items():
            if src == activity:
                total_correct += perf.correct_count
                total_count += perf.total_count

        dname = display(activity)
        if total_count > 0:
            acc = total_correct / total_count
            fill = ppm.get_color_for_accuracy(acc)
            label = f"{dname}\n({total_correct}/{total_count})"
        else:
            fill = "white"
            label = dname

        # Highlight nodes on the case path
        if activity in path_nodes:
            border_color = "#1f77b4" if activity in prefix_nodes else "#888888"
            dot.node(
                activity,
                label=label,
                fillcolor=fill,
                color=border_color,
                penwidth="3",
            )
        else:
            dot.node(activity, label=label, fillcolor=fill)

    # Start node
    start_activities = ppm.get_start_activities()
    total_cases = sum(start_activities.values())
    dot.node(
        "__START__",
        label=f"Start\n({total_cases})",
        shape="circle",
        fillcolor="#90EE90",
        style="filled",
    )
    for act, cnt in start_activities.items():
        if cnt >= 1:
            dot.edge(
                "__START__",
                act,
                label=str(cnt),
                color="#888888",
                penwidth=str(1 + np.log1p(cnt)),
            )

    # Edges
    for (src, tgt), dfg_count in ppm.dfg.items():
        if dfg_count < 1:
            continue

        key = (src, tgt)
        if key in ppm.transition_performance:
            perf = ppm.transition_performance[key]
            acc = perf.accuracy
            base_color = ppm.get_color_for_accuracy(acc)
            label = f"{perf.correct_count}/{perf.total_count}"
            pw = 1 + np.log1p(perf.total_count)
        else:
            base_color = "gray"
            label = f"0/{dfg_count}"
            pw = 1

        on_path = key in path_transitions
        on_prefix = key in prefix_transitions

        if on_path:
            # Highlighted edge — thicker, blue/dark overlay
            edge_color = "#1f77b4" if on_prefix else "#555555"
            dot.edge(
                src,
                tgt,
                label=label,
                color=edge_color,
                penwidth=str(max(pw, 2.5) + 2),
                style="bold",
            )
        else:
            # Faded background edge
            dot.edge(
                src,
                tgt,
                label=label,
                color=base_color + "55",  # add alpha for fading
                fontcolor="#aaaaaa",
                penwidth=str(pw),
            )

    # Legend
    with dot.subgraph(name="cluster_legend") as leg:
        leg.attr(label="Legend", style="dashed")
        leg.node("high_acc", "High Accuracy", fillcolor=ppm.get_color_for_accuracy(0.9))
        leg.node("mid_acc", "Medium Accuracy", fillcolor=ppm.get_color_for_accuracy(0.5))
        leg.node("low_acc", "Low Accuracy", fillcolor=ppm.get_color_for_accuracy(0.1))
        leg.node("case_path", "Case path", fillcolor="white", color="#1f77b4", penwidth="3")
        leg.edge("high_acc", "mid_acc", style="invis")
        leg.edge("mid_acc", "low_acc", style="invis")
        leg.edge("low_acc", "case_path", style="invis")

    return dot.pipe(format="svg").decode("utf-8")


@app.get("/api/models/{name}/process-map")
async def process_map(
    name: str,
    case_name: str = Query(..., description="Case to highlight"),
    prefix_length: Optional[int] = Query(None, description="Prefix length (highlights prefix vs suffix)"),
):
    """Return process performance map SVG with the case path highlighted."""
    if name not in get_available_model_names():
        raise HTTPException(404, f"Model '{name}' not found")

    def _build():
        config = get_model_config(name)
        ppm = _get_ppm(name)

        # Get the case's activity sequence
        cat_info = _manager.get_categorical_info(name)
        cat_names = [c.name for c in cat_info]
        num_names = [n.name for n in _manager.get_numerical_info(name)]
        editor = CaseEditor(cat_names, num_names)
        case_data = _manager.get_case_data(name, case_name)
        editable = editor.tensor_to_editable(case_data, activity_feature=config.concept_name)

        _, idx_to_activity = _manager.get_activity_mapping(name)
        activities = editor.get_activity_sequence(editable, config.concept_name, idx_to_activity)

        pl = prefix_length if prefix_length is not None else len(activities)
        return _build_highlighted_svg(ppm, activities, pl)

    try:
        svg = await run_in_threadpool(_build)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return Response(content=svg, media_type="image/svg+xml")
