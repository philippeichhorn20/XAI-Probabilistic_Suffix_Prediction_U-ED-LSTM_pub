"""
Interactive Pathway Explorer - Dash + Cytoscape Application

Click activity nodes in the DFG graph to build a pathway step by step,
then analyze feature effects at decision points.

Run with: python src/interpretability/interactive_tool/pathway_explorer.py
Opens at: http://127.0.0.1:8050
"""

import sys
import warnings
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

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, callback, Input, Output, State, no_update, dash_table
import dash_cytoscape as cyto

from src.interpretability.interactive_tool.config import (
    get_available_model_names,
    get_available_model_display_names,
    get_model_config,
)
from src.interpretability.interactive_tool.model_manager import ModelManager
from src.interpretability.utils.tensor_decoder import TensorDecoder
from src.interpretability.perturbation_methods.revised_plus.revised_plus import (
    RevisedPlusModelPredictor,
)
from src.interpretability.interactive_tool.dfg_analyzer import DFGAnalyzer

# Load dagre layout extension
cyto.load_extra_layouts()

warnings.filterwarnings('ignore')

# ============================================================
# Server-side cache (single-user research tool)
# ============================================================

_cache = {}  # model_name -> {model, dataset, decoder, analyzer, dfg_result, predictor, config}
_manager = ModelManager(PROJECT_ROOT)


def _load_model_cache(model_name):
    """Lazy-load all heavy objects for a model."""
    if model_name in _cache:
        return _cache[model_name]

    config = get_model_config(model_name)
    model = _manager.load_model(model_name)
    dataset = _manager.load_test_dataset(model_name)
    decoder = TensorDecoder(dataset)
    csv_path = config.get_csv_path(PROJECT_ROOT)
    analyzer = DFGAnalyzer(
        dataset, config.concept_name, decoder,
        csv_path=str(csv_path) if csv_path else None,
        csv_case_col=config.csv_case_col,
    )
    dfg_result = analyzer.build_dfg()
    predictor = RevisedPlusModelPredictor(
        model, suffix_step=0, activity_feature=config.concept_name,
    )
    entry = {
        'model': model,
        'dataset': dataset,
        'decoder': decoder,
        'analyzer': analyzer,
        'dfg_result': dfg_result,
        'predictor': predictor,
        'config': config,
    }
    _cache[model_name] = entry
    return entry


# ============================================================
# Cytoscape element and stylesheet builders
# ============================================================

START_NODE_ID = '__START__'


def build_cytoscape_elements(dfg_result):
    """Build Cytoscape elements (nodes + edges) from a DFGResult.

    Adds a synthetic START node connected to every activity that appears
    as the first event in a trace, with the edge label showing how many
    traces start with that activity.
    """
    all_activities = set()
    for (src, tgt) in dfg_result.dfg:
        all_activities.add(src)
        all_activities.add(tgt)

    # Count how many traces start with each activity (single-element pathways)
    start_counts = {}
    for pathway, cases in dfg_result.pathway_index.items():
        if len(pathway) == 1:
            start_counts[pathway[0]] = len(cases)

    nodes = []

    # START node
    nodes.append({
        'data': {'id': START_NODE_ID, 'label': 'START'},
        'classes': 'start-end',
    })

    for act in sorted(all_activities):
        cls = 'decision' if act in dfg_result.decision_points else 'normal'
        nodes.append({
            'data': {'id': act, 'label': act},
            'classes': cls,
        })

    edges = []

    # Edges from START to every first-in-trace activity
    for act in sorted(start_counts):
        count = start_counts[act]
        edges.append({
            'data': {
                'source': START_NODE_ID,
                'target': act,
                'count': count,
                'label': str(count),
            },
            'classes': 'start-end-edge',
        })

    # Real DFG edges
    for (src, tgt), count in dfg_result.dfg.items():
        edges.append({
            'data': {
                'source': src,
                'target': tgt,
                'count': count,
                'label': str(count),
            },
        })

    return nodes + edges


BASE_STYLESHEET = [
    # Base node style
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'text-wrap': 'wrap',
            'text-max-width': '120px',
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#FFFFFF',
            'border-width': 1.5,
            'border-color': '#999',
            'shape': 'round-rectangle',
            'width': 'label',
            'height': 'label',
            'padding': '10px',
        },
    },
    # Decision point nodes
    {
        'selector': '.decision',
        'style': {
            'background-color': '#FFD700',
            'border-width': 2.5,
            'border-color': '#B8860B',
        },
    },
    # START / END mock nodes
    {
        'selector': '.start-end',
        'style': {
            'background-color': '#555',
            'color': '#FFFFFF',
            'shape': 'ellipse',
            'border-width': 0,
            'font-weight': 'bold',
            'font-size': '12px',
            'width': '60px',
            'height': '30px',
        },
    },
    # START / END edges (dashed, no label)
    {
        'selector': '.start-end-edge',
        'style': {
            'line-style': 'dashed',
            'line-color': '#999',
            'target-arrow-color': '#999',
            'label': '',
        },
    },
    # Base edge style
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#999',
            'line-color': '#CCC',
            'width': 1.5,
            'label': 'data(label)',
            'font-size': '9px',
            'text-rotation': 'autorotate',
            'color': '#888',
        },
    },
    # Pathway node (blue)
    {
        'selector': '.pathway-node',
        'style': {
            'background-color': '#4A90D9',
            'border-color': '#2C5F9A',
            'border-width': 3,
            'color': '#FFFFFF',
        },
    },
    # Successor node (green dashed border)
    {
        'selector': '.successor-node',
        'style': {
            'border-color': '#2ECC71',
            'border-width': 3,
            'border-style': 'dashed',
        },
    },
    # Pathway edge (blue, thicker)
    {
        'selector': '.pathway-edge',
        'style': {
            'line-color': '#4A90D9',
            'target-arrow-color': '#4A90D9',
            'width': 3,
        },
    },
]


def compute_node_classes(pathway, successors_map):
    """Compute per-node CSS classes based on current pathway."""
    classes = {}
    if not pathway:
        return classes

    for act in pathway:
        classes[act] = 'pathway-node'

    last_act = pathway[-1]
    successors = successors_map.get(last_act, {})
    for succ in successors:
        if succ not in classes:
            classes[succ] = 'successor-node'

    return classes


def apply_classes_to_elements(elements, node_classes, pathway):
    """Return updated elements with classes applied."""
    pathway_edges = set()
    for i in range(len(pathway) - 1):
        pathway_edges.add((pathway[i], pathway[i + 1]))

    updated = []
    for el in elements:
        el = dict(el)
        el['data'] = dict(el['data'])
        if 'source' not in el['data']:
            # Node
            node_id = el['data']['id']
            base_cls = 'decision' if el.get('classes', '').startswith('decision') else 'normal'
            if 'decision' in el.get('classes', ''):
                base_cls = 'decision'
            extra = node_classes.get(node_id, '')
            el['classes'] = f'{base_cls} {extra}'.strip()
        else:
            # Edge
            src = el['data']['source']
            tgt = el['data']['target']
            if (src, tgt) in pathway_edges:
                el['classes'] = 'pathway-edge'
            else:
                el['classes'] = ''
        updated.append(el)
    return updated


# ============================================================
# Plotly chart builders (same logic as before)
# ============================================================

def create_prediction_bar_chart(probs, activity_names, successors):
    """Horizontal bar chart of P(next_activity). Red = observed branch, teal = other."""
    nonzero_mask = probs > 0.005
    indices = [i for i in range(len(probs)) if nonzero_mask[i]]
    labels = [activity_names[i] for i in indices]
    values = [float(probs[i]) for i in indices]

    sorted_pairs = sorted(zip(labels, values), key=lambda x: -x[1])
    labels = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]
    colors = ['#FF6B6B' if lbl in successors else '#4ECDC4' for lbl in labels]

    fig = px.bar(
        x=values, y=labels, orientation='h',
        labels={'x': 'Probability', 'y': 'Activity'},
    )
    fig.update_traces(marker_color=colors)
    fig.update_layout(
        height=max(200, len(labels) * 25),
        showlegend=False,
        yaxis=dict(autorange='reversed'),
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text='P(next activity) — red = observed branch, teal = other', font_size=11),
    )
    return fig


def create_branch_top3_chart(df_sweep, branch, b_idx, baseline_p):
    """Bar chart of top-3 feature values maximizing P(branch)."""
    df_all = df_sweep.copy()
    df_all['prob_branch'] = df_all['probability_vector'].apply(lambda pv: pv[b_idx])
    top3 = df_all.nlargest(3, 'prob_branch')

    labels = [f"{row['feature']} = {row['value_name']}" for _, row in top3.iterrows()]
    probs = top3['prob_branch'].tolist()

    fig = px.bar(x=labels, y=probs, labels={'x': '', 'y': 'P(branch)'})
    fig.update_traces(marker_color='#4ECDC4')
    fig.add_hline(
        y=baseline_p, line_dash='dash', line_color='red',
        annotation_text=f'baseline ({baseline_p:.3f})',
        annotation_position='top right',
    )
    fig.update_layout(
        title=dict(text=f'Branch: {branch}', font_size=11),
        height=300,
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_tickangle=-30,
    )
    return fig


def create_heatmap(df_sweep, feature_idx, feature_name, is_case_level, branches, branch_indices):
    """Plotly heatmap: feature values x branches, with original value annotated."""
    df_feat = df_sweep[df_sweep['feature_idx'] == feature_idx].copy()

    if not is_case_level:
        last_pos = df_feat['position'].max()
        df_feat = df_feat[df_feat['position'] == last_pos]

    heatmap_data = {}
    for branch, b_idx in zip(branches, branch_indices):
        heatmap_data[branch] = df_feat['probability_vector'].apply(lambda pv: pv[b_idx]).values

    value_labels = df_feat['value_name'].tolist()
    original_mask = df_feat['is_original'].tolist()
    hm_df = pd.DataFrame(heatmap_data, index=value_labels)

    hm_df['_var'] = hm_df.var(axis=1)
    hm_df = hm_df.sort_values('_var', ascending=False)
    hm_df = hm_df.drop(columns='_var')

    MAX_ROWS = 40
    title_suffix = ''
    total_rows = len(hm_df)
    if total_rows > MAX_ROWS:
        top_rows = hm_df.head(MAX_ROWS - 1)
        orig_values = [vl for vl, om in zip(value_labels, original_mask) if om]
        for ov in orig_values:
            if ov in hm_df.index and ov not in top_rows.index:
                top_rows = pd.concat([top_rows, hm_df.loc[[ov]]])
        title_suffix = f' (top {len(top_rows)}/{total_rows} by variance)'
        hm_df = top_rows

    orig_label = None
    for vl, om in zip(value_labels, original_mask):
        if om:
            orig_label = vl
            break

    annot_text = []
    for row_label in hm_df.index:
        row_vals = []
        for col in hm_df.columns:
            val = hm_df.loc[row_label, col]
            txt = f'{val:.2f}'
            if row_label == orig_label:
                txt += ' *'
            row_vals.append(txt)
        annot_text.append(row_vals)

    pos_label = ' (case-level)' if is_case_level else ' (last position)'
    fig = px.imshow(
        hm_df.values,
        x=list(hm_df.columns),
        y=list(hm_df.index),
        color_continuous_scale='YlOrRd',
        zmin=0, zmax=1,
        labels=dict(x='Branch', y=f'{feature_name} values', color='P(branch)'),
    )
    if len(hm_df) <= 40:
        fig.update_traces(
            text=annot_text,
            texttemplate='%{text}',
            textfont_size=9,
        )
    fig.update_layout(
        title=dict(
            text=f'{feature_name}{pos_label} — P(branch) under substitution{title_suffix}<br>'
                 f'<span style="font-size:10px">(* = original value)</span>',
            font_size=12,
        ),
        height=max(300, len(hm_df) * 22 + 100),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


# ============================================================
# Dash App
# ============================================================

app = Dash(__name__)
app.title = "Pathway Explorer"

model_names = get_available_model_names()
display_names = get_available_model_display_names()
model_options = [{'label': display_names.get(n, n), 'value': n} for n in model_names]

app.layout = html.Div([
    # Stores
    dcc.Store(id='pathway-store', data=[]),
    dcc.Store(id='model-store', data=''),
    dcc.Store(id='elements-store', data=[]),

    # Header row
    html.Div([
        html.Div([
            html.Label('Model:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
            dcc.Dropdown(
                id='model-dropdown',
                options=model_options,
                value=model_names[0] if model_names else None,
                clearable=False,
                style={'width': '250px', 'display': 'inline-block', 'verticalAlign': 'middle'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Button('Clear Pathway', id='clear-btn', n_clicks=0,
                     style={'marginLeft': 'auto', 'padding': '6px 16px'}),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'padding': '10px 20px',
        'borderBottom': '1px solid #ddd', 'backgroundColor': '#f8f9fa',
    }),

    # Loading indicator for model
    dcc.Loading(
        id='loading-model',
        type='circle',
        children=[html.Div(id='model-load-signal', style={'display': 'none'})],
    ),

    # Cytoscape graph
    html.Div([
        cyto.Cytoscape(
            id='dfg-graph',
            elements=[],
            stylesheet=BASE_STYLESHEET,
            layout={'name': 'dagre', 'rankDir': 'LR', 'nodeSep': 50, 'rankSep': 100},
            style={'width': '100%', 'height': '500px', 'border': '1px solid #ddd'},
            boxSelectionEnabled=False,
            autounselectify=True,
            userZoomingEnabled=True,
            userPanningEnabled=True,
        ),
    ], style={'padding': '10px 20px'}),

    # Pathway info bar
    html.Div([
        html.Div([
            html.Span('Pathway: ', style={'fontWeight': 'bold'}),
            html.Span(id='pathway-display', children='(click a node to start)'),
        ], style={'flex': '1'}),
        html.Div(id='matching-cases-display', style={'marginLeft': '20px', 'fontWeight': 'bold'}),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'padding': '8px 20px',
        'backgroundColor': '#eef2f7', 'borderTop': '1px solid #ddd',
        'borderBottom': '1px solid #ddd',
    }),

    # Case selection + analyze
    html.Div([
        html.Div([
            html.Label('Case:', style={'fontWeight': 'bold', 'marginRight': '8px'}),
            dcc.Dropdown(
                id='case-dropdown',
                options=[],
                value=None,
                placeholder='Select a case...',
                style={'width': '300px', 'display': 'inline-block', 'verticalAlign': 'middle'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Button('Analyze', id='analyze-btn', n_clicks=0,
                     style={'marginLeft': '16px', 'padding': '6px 20px',
                            'backgroundColor': '#4A90D9', 'color': 'white',
                            'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
    ], style={
        'display': 'flex', 'alignItems': 'center', 'padding': '10px 20px',
        'borderBottom': '1px solid #ddd',
    }),

    # Analysis results
    dcc.Loading(
        id='loading-analysis',
        type='default',
        children=[html.Div(id='analysis-results')],
    ),

], style={'fontFamily': 'Arial, sans-serif', 'fontSize': '13px'})


# ============================================================
# Callbacks
# ============================================================

@callback(
    Output('dfg-graph', 'elements'),
    Output('elements-store', 'data'),
    Output('pathway-store', 'data'),
    Output('model-store', 'data'),
    Output('model-load-signal', 'children'),
    Input('model-dropdown', 'value'),
)
def on_model_change(model_name):
    """Load model + DFG, build cytoscape elements, clear pathway."""
    if not model_name:
        return [], [], [], '', ''

    entry = _load_model_cache(model_name)
    elements = build_cytoscape_elements(entry['dfg_result'])
    return elements, elements, [], model_name, ''


@callback(
    Output('dfg-graph', 'elements', allow_duplicate=True),
    Output('pathway-store', 'data', allow_duplicate=True),
    Output('pathway-display', 'children', allow_duplicate=True),
    Output('matching-cases-display', 'children', allow_duplicate=True),
    Output('case-dropdown', 'options', allow_duplicate=True),
    Output('case-dropdown', 'value', allow_duplicate=True),
    Input('dfg-graph', 'tapNodeData'),
    State('pathway-store', 'data'),
    State('model-store', 'data'),
    State('elements-store', 'data'),
    prevent_initial_call=True,
)
def on_node_tap(tap_data, pathway, model_name, base_elements):
    """Handle node click: extend or start pathway."""
    if tap_data is None or not model_name:
        return no_update, no_update, no_update, no_update, no_update, no_update

    tapped = tap_data['id']

    # Ignore clicks on synthetic START/END nodes
    if tapped == START_NODE_ID:
        return no_update, no_update, no_update, no_update, no_update, no_update
    entry = _load_model_cache(model_name)
    dfg_result = entry['dfg_result']
    successors_map = dfg_result.successors_map

    if pathway is None:
        pathway = []

    if pathway:
        last_act = pathway[-1]
        successors = successors_map.get(last_act, {})
        if tapped in successors:
            pathway = pathway + [tapped]
        else:
            pathway = [tapped]
    else:
        pathway = [tapped]

    # Compute display
    node_classes = compute_node_classes(pathway, successors_map)
    updated_elements = apply_classes_to_elements(base_elements, node_classes, pathway)
    pathway_text = ' \u2192 '.join(pathway)

    # Matching cases
    pathway_tuple = tuple(pathway)
    matching = dfg_result.pathway_index.get(pathway_tuple, [])
    match_text = f'Matching cases: {len(matching)}'

    # Case dropdown options
    case_options = []
    for i, (ds_idx, trace_len) in enumerate(matching):
        case_id = entry['dataset'][ds_idx][2]
        case_options.append({'label': f'{case_id}', 'value': str(i)})

    return updated_elements, pathway, pathway_text, match_text, case_options, None


@callback(
    Output('dfg-graph', 'elements', allow_duplicate=True),
    Output('pathway-store', 'data', allow_duplicate=True),
    Output('pathway-display', 'children', allow_duplicate=True),
    Output('matching-cases-display', 'children', allow_duplicate=True),
    Output('case-dropdown', 'options', allow_duplicate=True),
    Output('case-dropdown', 'value', allow_duplicate=True),
    Output('analysis-results', 'children', allow_duplicate=True),
    Input('clear-btn', 'n_clicks'),
    State('elements-store', 'data'),
    prevent_initial_call=True,
)
def on_clear(n_clicks, base_elements):
    """Reset pathway to empty."""
    if not base_elements:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # Restore base elements (no pathway/successor classes)
    restored = apply_classes_to_elements(base_elements, {}, [])
    return restored, [], '(click a node to start)', '', [], None, None


@callback(
    Output('analysis-results', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('pathway-store', 'data'),
    State('case-dropdown', 'value'),
    State('model-store', 'data'),
    prevent_initial_call=True,
)
def on_analyze(n_clicks, pathway, case_value, model_name):
    """Run prediction + feature sweep for selected pathway and case."""
    if not pathway or case_value is None or not model_name:
        return html.Div('Select a pathway and case first.', style={'padding': '20px', 'color': '#999'})

    case_index = int(case_value)
    entry = _load_model_cache(model_name)
    dfg_result = entry['dfg_result']
    analyzer = entry['analyzer']
    predictor = entry['predictor']
    config = entry['config']
    model = entry['model']

    pathway_tuple = tuple(pathway)

    # Build context (baseline prediction)
    context = analyzer.build_pathway_context(dfg_result, pathway_tuple, case_index, model)

    # Run feature sweep
    sweep_config = analyzer.get_sweepable_features(config.case_level_cat)
    sweep_result = analyzer.run_sweep(context, sweep_config, predictor)

    # --- Build result layout ---
    children = []

    # Section: Prediction
    children.append(html.H3(
        f"Prediction: {context.original_pred_name} (p={context.original_pred_prob:.3f})",
        style={'padding': '10px 20px 0 20px'},
    ))

    # Prefix/suffix tables + bar chart
    prefix_df = context.prefix_df
    suffix_df = context.suffix_df
    pred_chart = create_prediction_bar_chart(
        context.original_probs,
        dfg_result.activity_names,
        context.successors,
    )

    # Decision point info
    decision_info = []
    if context.is_decision:
        succs = context.successors
        total = sum(succs.values())
        for s, c in sorted(succs.items(), key=lambda x: -x[1]):
            decision_info.append(html.Div(f'{s}: {c}/{total} ({c/total*100:.1f}%)'))

    children.append(html.Div([
        html.Div([
            html.Div([
                html.Strong(f'Case: {context.case_id}'),
                html.Div(style={'height': '8px'}),
                html.Strong('Observed branches:') if decision_info else html.Span(),
                *decision_info,
                html.Div(style={'height': '12px'}),
                html.Strong('Prefix:'),
                dash_table.DataTable(
                    data=prefix_df.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in prefix_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '250px', 'overflowY': 'auto'},
                    style_cell={'fontSize': '11px', 'padding': '4px 8px', 'textAlign': 'left'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                ),
                html.Div(style={'height': '12px'}),
                html.Strong('Actual suffix:'),
                dash_table.DataTable(
                    data=suffix_df.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in suffix_df.columns],
                    style_table={'overflowX': 'auto', 'maxHeight': '250px', 'overflowY': 'auto'},
                    style_cell={'fontSize': '11px', 'padding': '4px 8px', 'textAlign': 'left'},
                    style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                ) if not suffix_df.empty else html.Div(
                    '(no suffix — trace ends here)',
                    style={'color': '#999', 'fontStyle': 'italic', 'marginTop': '4px'},
                ),
            ], style={'flex': '1', 'padding': '10px'}),
            html.Div([
                dcc.Graph(figure=pred_chart, config={'displayModeBar': False}),
            ], style={'flex': '1', 'padding': '10px'}),
        ], style={'display': 'flex'}),
    ], style={'padding': '0 20px'}))

    # Section: Feature Sweep
    df_sweep = sweep_result.df_sweep
    n_preds = len(df_sweep)
    n_features = df_sweep['feature'].nunique()
    n_flips = int((~df_sweep['is_original'] & (df_sweep['predicted_class'] != sweep_result.original_pred_idx)).sum())

    children.append(html.Hr())
    children.append(html.H3(
        f'Feature Sweep: {n_preds} predictions in {sweep_result.elapsed:.2f}s '
        f'({n_features} features, {n_flips} flips)',
        style={'padding': '0 20px'},
    ))

    # Branch top-3 charts
    branches = sweep_result.branches
    branch_indices = sweep_result.branch_indices

    if branches:
        branch_charts = []
        for branch, b_idx in zip(branches, branch_indices):
            baseline_p = float(sweep_result.original_probs[b_idx])
            fig = create_branch_top3_chart(df_sweep, branch, b_idx, baseline_p)
            branch_charts.append(
                html.Div([
                    dcc.Graph(figure=fig, config={'displayModeBar': False}),
                ], style={'flex': '1', 'minWidth': '300px'}),
            )
        children.append(html.Div(
            branch_charts,
            style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '0 20px'},
        ))

    # Per-feature heatmaps
    children.append(html.H4('Feature Heatmaps', style={'padding': '10px 20px 0 20px'}))
    swept_features = df_sweep[['feature_idx', 'feature', 'is_case_level']].drop_duplicates()
    for _, row in swept_features.iterrows():
        fig = create_heatmap(
            df_sweep,
            int(row['feature_idx']),
            row['feature'],
            bool(row['is_case_level']),
            branches,
            branch_indices,
        )
        children.append(html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
        ], style={'padding': '0 20px'}))

    return html.Div(children)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
