"""
Process Performance Map - Visualize model prediction performance on a process map.

This module combines process mining (pm4py) with model prediction analysis to create
a process map where transitions are colored based on prediction accuracy.

Usage:
    from src.interpretability.process_performance_map import ProcessPerformanceMap

    ppm = ProcessPerformanceMap(model, test_dataset, eval_helper)
    ppm.collect_predictions()
    ppm.visualize()
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

try:
    import pm4py
    from pm4py.objects.log.obj import EventLog, Trace, Event
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
    from pm4py.visualization.dfg import visualizer as dfg_visualization
    from pm4py.statistics.start_activities.log import get as get_start_activities
    from pm4py.statistics.end_activities.log import get as get_end_activities
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    warnings.warn("pm4py not installed. Install with: pip install pm4py")

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


@dataclass
class PredictionRecord:
    """Single prediction record."""
    case_id: str
    prefix_length: int
    current_activity: str
    actual_next_activity: str
    predicted_next_activity: str
    correct: bool
    confidence: float  # Probability of predicted class
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionPerformance:
    """Performance metrics for a single transition (A -> B)."""
    source: str
    target: str
    correct_count: int = 0
    incorrect_count: int = 0
    predictions: List[PredictionRecord] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return self.correct_count + self.incorrect_count

    @property
    def accuracy(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count

    def get_feature_breakdown(self, feature_name: str) -> Dict[str, Dict[str, int]]:
        """
        Get performance breakdown by a specific feature.

        Returns:
            Dict mapping feature values to {correct: int, incorrect: int}
        """
        breakdown = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        for pred in self.predictions:
            if feature_name in pred.features:
                val = pred.features[feature_name]
                if pred.correct:
                    breakdown[val]['correct'] += 1
                else:
                    breakdown[val]['incorrect'] += 1
        return dict(breakdown)


class ProcessPerformanceMap:
    """
    Visualize model prediction performance overlaid on a process map.

    This class:
    1. Collects predictions from the model for all prefixes in test data
    2. Mines a process map (DFG) from the test data
    3. Colors transitions based on prediction accuracy
    4. Provides interactive exploration of performance by features
    """

    def __init__(
        self,
        model,
        test_dataset,
        eval_helper,
        activity_key: str = 'Activity',
        device: Optional[str] = None
    ):
        """
        Initialize the ProcessPerformanceMap.

        Args:
            model: The trained DropoutUncertaintyEncoderDecoderLSTM model
            test_dataset: Test dataset (torch Dataset)
            eval_helper: Evaluation helper with case iteration logic
            activity_key: Name of the activity feature
            device: Device for model inference
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("pm4py is required. Install with: pip install pm4py")

        self.model = model
        self.test_dataset = test_dataset
        self.eval_helper = eval_helper
        self.activity_key = activity_key
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Extract feature info from model
        self.data_set_categories = model.data_set_categories
        self._extract_feature_info()

        # Storage for predictions and performance
        self.predictions: List[PredictionRecord] = []
        self.transition_performance: Dict[Tuple[str, str], TransitionPerformance] = {}
        self.activity_labels: Dict[int, str] = {}

        # Process map data
        self.event_log: Optional[EventLog] = None
        self.dfg: Optional[Dict] = None

    def _extract_feature_info(self):
        """Extract feature information from model."""
        cat_categories, num_categories = self.data_set_categories

        # Get activity label mapping
        for name, size, label_dict in cat_categories:
            if name == self.activity_key:
                self.activity_labels = {v: k for k, v in label_dict.items()}
                break

        # Get all categorical feature names and their mappings
        self.categorical_features = {}
        for name, size, label_dict in cat_categories:
            self.categorical_features[name] = {v: k for k, v in label_dict.items()}

        # Get numerical feature names
        self.numerical_features = [name for name, _, _ in num_categories]

    def get_activity_label_dict(self) -> Dict:
        """
        Get the raw activity label dictionary from the model.

        Returns:
            Dict mapping activity names to indices (as stored in model)
        """
        cat_categories, _ = self.data_set_categories
        for name, size, label_dict in cat_categories:
            if name == self.activity_key:
                return label_dict
        return {}

    def get_activity_name_map(self) -> Dict[str, str]:
        """
        Get a mapping from activity IDs (as strings) to display names.

        This can be passed directly to visualize(activity_name_map=...).
        Note: If the label_dict already has meaningful names, this returns
        a mapping where keys equal values.

        Returns:
            Dict[str, str] mapping activity ID strings to display names
        """
        return {str(idx): name for idx, name in self.activity_labels.items()}

    def collect_predictions(self, show_progress: bool = True, case_names: list = None) -> None:
        """
        Collect predictions for all prefixes in the test data.

        Args:
            show_progress: Whether to show progress bar
            case_names: Optional list of case names to process. If None, process all cases.
        """
        self.model.eval()
        self.model.to(self.device)

        if case_names is not None:
            cases = [(name, self.eval_helper.cases[name]) for name in case_names if name in self.eval_helper.cases]
        else:
            cases = list(self.eval_helper.cases.items())

        if show_progress:
            try:
                from tqdm import tqdm
                cases = tqdm(cases, desc="Collecting predictions")
            except ImportError:
                pass

        for case_name, case_data in cases:
            self._collect_case_predictions(case_name, case_data)

        # Aggregate into transition performance
        self._aggregate_transition_performance()

        print(f"Collected {len(self.predictions)} predictions")
        print(f"Found {len(self.transition_performance)} unique transitions")

    def _collect_case_predictions(self, case_name: str, case_data) -> None:
        """Collect predictions for all prefixes of a single case."""

        for prefix_len, prefix, suffix in self.eval_helper._iterate_case(case_data):
            # Skip if no suffix (nothing to predict)
            if suffix is None:
                continue

            # Get actual next activity from suffix
            suffix_cats, suffix_nums = suffix
            if not suffix_cats or len(suffix_cats[0]) == 0:
                continue

            # First activity in suffix is the next activity to predict
            actual_next_idx = suffix_cats[0][0, 0].item()  # First cat feature, first position
            actual_next_activity = self.activity_labels.get(actual_next_idx, str(actual_next_idx))

            # Get current activity (last in prefix)
            prefix_cats, prefix_nums = prefix
            current_idx = prefix_cats[0][0, -1].item()  # Last position in prefix
            current_activity = self.activity_labels.get(current_idx, str(current_idx))

            # Skip padding (activity index 0 is usually padding)
            if current_idx == 0:
                continue

            # Get model prediction
            with torch.no_grad():
                prefix_device = (
                    [t.to(self.device) for t in prefix_cats],
                    [t.to(self.device) for t in prefix_nums]
                )
                predictions, _, _, _ = self.model(prefix_device)

            cat_preds, num_preds = predictions
            activity_logits = cat_preds[f'{self.activity_key}_mean']

            # Get prediction for next step (step 0)
            if activity_logits.dim() == 3:
                activity_logits = activity_logits[0]  # First suffix step

            probs = torch.softmax(activity_logits, dim=-1)
            predicted_idx = probs.argmax(dim=-1).item()
            confidence = probs[0, predicted_idx].item()
            predicted_activity = self.activity_labels.get(predicted_idx, str(predicted_idx))

            # Check correctness
            correct = (predicted_idx == actual_next_idx)

            # Extract feature values for this prefix
            features = self._extract_features(prefix_cats, prefix_nums, prefix_len)

            # Create prediction record
            record = PredictionRecord(
                case_id=case_name,
                prefix_length=prefix_len,
                current_activity=current_activity,
                actual_next_activity=actual_next_activity,
                predicted_next_activity=predicted_activity,
                correct=correct,
                confidence=confidence,
                features=features
            )

            self.predictions.append(record)

    def _extract_features(
        self,
        prefix_cats: List[torch.Tensor],
        prefix_nums: List[torch.Tensor],
        prefix_len: int
    ) -> Dict[str, Any]:
        """Extract feature values from the prefix.

        For event-level features (Activity, Resource): use last position.
        For case-level features (Variant index, customer, etc.): use first non-padding value.
        """
        features = {}

        cat_categories, num_categories = self.data_set_categories

        # Features that are case-level (constant across all events)
        # These should take first non-padding value, not last position
        case_level_features = {
            'Variant index', 'VariantIndex', 'customer', 'product',
            'responsible_section', 'seriousness', 'seriousness_2',
            'service_level', 'service_type', 'support_section', 'workgroup',
            'case:LoanGoal', 'case:ApplicationType'
        }

        # Extract categorical features
        for i, (name, _, label_dict) in enumerate(cat_categories):
            if i < len(prefix_cats):
                tensor = prefix_cats[i]
                if tensor.dim() > 1:
                    tensor = tensor.squeeze(0)

                label_map = {v: k for k, v in label_dict.items()}

                if name in case_level_features:
                    # For case-level features, find first non-padding value
                    idx = 0
                    for val in tensor.tolist():
                        if val != 0:
                            idx = val
                            break
                else:
                    # For event-level features, use last position
                    idx = tensor[-1].item()

                features[name] = label_map.get(idx, str(idx))

        # Extract numerical features (use last position)
        for i, (name, _, _) in enumerate(num_categories):
            if i < len(prefix_nums):
                features[name] = prefix_nums[i][0, -1].item()

        # Add prefix length as a feature
        features['prefix_length'] = prefix_len

        return features

    def _aggregate_transition_performance(self) -> None:
        """Aggregate predictions into transition-level performance metrics."""
        self.transition_performance = {}

        for pred in self.predictions:
            # Key is (current_activity, actual_next_activity)
            # This represents the TRUE transition in the process
            key = (pred.current_activity, pred.actual_next_activity)

            if key not in self.transition_performance:
                self.transition_performance[key] = TransitionPerformance(
                    source=pred.current_activity,
                    target=pred.actual_next_activity
                )

            tp = self.transition_performance[key]
            tp.predictions.append(pred)

            if pred.correct:
                tp.correct_count += 1
            else:
                tp.incorrect_count += 1

    def build_event_log(self) -> EventLog:
        """
        Build a pm4py EventLog from the test data for process mining.

        Returns:
            pm4py EventLog object
        """
        log = EventLog()

        for case_name, case_data in self.eval_helper.cases.items():
            trace = Trace()
            trace.attributes['concept:name'] = case_name

            # case_data is (cat_tensors, num_tensors, case_name) tuple from dataset
            # Access cat_tensors as case_data[0]
            cat_tensors = case_data[0]

            # Get activity indices
            activity_tensor = cat_tensors[0]  # First categorical is Activity
            if activity_tensor.dim() > 1:
                activity_tensor = activity_tensor.squeeze(0)

            for i, act_idx in enumerate(activity_tensor.tolist()):
                if act_idx == 0:  # Skip padding
                    continue

                activity_name = self.activity_labels.get(act_idx, str(act_idx))

                event = Event()
                event['concept:name'] = activity_name
                event['time:timestamp'] = pd.Timestamp('2020-01-01') + pd.Timedelta(hours=i)
                trace.append(event)

            if len(trace) > 0:
                log.append(trace)

        self.event_log = log
        return log

    def mine_dfg(self) -> Dict[Tuple[str, str], int]:
        """
        Mine a Directly-Follows Graph from the event log.

        Returns:
            DFG as dict mapping (source, target) to frequency
        """
        if self.event_log is None:
            self.build_event_log()

        self.dfg = dfg_discovery.apply(self.event_log)
        return self.dfg

    def get_start_activities(self) -> Dict[str, int]:
        """
        Get the start activities and their frequencies.

        Returns:
            Dict mapping activity name to count of cases starting with it
        """
        start_counts: Dict[str, int] = {}

        for case_name, case_data in self.eval_helper.cases.items():
            cat_tensors = case_data[0]
            activity_tensor = cat_tensors[0]
            if activity_tensor.dim() > 1:
                activity_tensor = activity_tensor.squeeze(0)

            # Find first non-padding activity
            for act_idx in activity_tensor.tolist():
                if act_idx != 0:
                    activity_name = self.activity_labels.get(act_idx, str(act_idx))
                    start_counts[activity_name] = start_counts.get(activity_name, 0) + 1
                    break

        return start_counts

    def get_color_for_accuracy(self, accuracy: float) -> str:
        """
        Get color for a given accuracy value.

        Args:
            accuracy: Value between 0 and 1

        Returns:
            Hex color string (red -> yellow -> green gradient)
        """
        # Red (low) -> Yellow (mid) -> Green (high)
        if accuracy < 0.5:
            # Red to Yellow
            r = 255
            g = int(255 * (accuracy * 2))
            b = 0
        else:
            # Yellow to Green
            r = int(255 * (1 - (accuracy - 0.5) * 2))
            g = 255
            b = 0

        return f'#{r:02x}{g:02x}{b:02x}'

    def visualize(
        self,
        min_edge_count: int = 1,
        show_counts: bool = True,
        figsize: Tuple[int, int] = (16, 12),
        filename: Optional[str] = None,
        activity_name_map: Optional[Dict[str, str]] = None,
        show_start_node: bool = True
    ) -> 'graphviz.Digraph':
        """
        Create a process map visualization with performance coloring.

        Args:
            min_edge_count: Minimum transition count to include
            show_counts: Whether to show correct/total counts on edges
            figsize: Figure size (not used for graphviz, kept for API consistency)
            filename: If provided, save the visualization to this file
            activity_name_map: Optional dict mapping activity IDs to display names
            show_start_node: Whether to show a fictive "Start" node

        Returns:
            Graphviz Digraph object
        """
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("graphviz is required. Install with: pip install graphviz")

        if self.dfg is None:
            self.mine_dfg()

        # Helper to get display name for an activity
        def get_display_name(activity_id: str) -> str:
            if activity_name_map and activity_id in activity_name_map:
                return activity_name_map[activity_id]
            return activity_id

        # Create graphviz digraph
        dot = graphviz.Digraph(comment='Process Performance Map')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='white')

        # Collect all activities
        activities: Set[str] = set()
        for (src, tgt), count in self.dfg.items():
            activities.add(src)
            activities.add(tgt)

        # Add activity nodes
        for activity in activities:
            # Calculate overall accuracy for this activity as source
            total_correct = 0
            total_count = 0
            for (src, tgt), perf in self.transition_performance.items():
                if src == activity:
                    total_correct += perf.correct_count
                    total_count += perf.total_count

            display_name = get_display_name(activity)
            if total_count > 0:
                node_accuracy = total_correct / total_count
                node_color = self.get_color_for_accuracy(node_accuracy)
                label = f'{display_name}\n({total_correct}/{total_count})'
            else:
                node_color = 'white'
                label = display_name

            dot.node(activity, label=label, fillcolor=node_color)

        # Add start node if requested
        if show_start_node:
            start_activities = self.get_start_activities()
            total_cases = sum(start_activities.values())

            # Add the Start node (circle shape, green)
            dot.node('__START__', label=f'Start\n({total_cases})',
                    shape='circle', fillcolor='#90EE90', style='filled')

            # Add edges from Start to each starting activity
            for activity, count in start_activities.items():
                if count >= min_edge_count:
                    dot.edge('__START__', activity, label=str(count),
                            color='#888888', penwidth=str(1 + np.log1p(count)))

        # Add edges with performance coloring
        for (src, tgt), dfg_count in self.dfg.items():
            if dfg_count < min_edge_count:
                continue

            key = (src, tgt)
            if key in self.transition_performance:
                perf = self.transition_performance[key]
                accuracy = perf.accuracy
                color = self.get_color_for_accuracy(accuracy)

                if show_counts:
                    label = f'{perf.correct_count}/{perf.total_count}'
                else:
                    label = f'{accuracy:.0%}'

                # Edge thickness based on count
                penwidth = str(1 + np.log1p(perf.total_count))
            else:
                # Transition exists in process but no predictions for it
                color = 'gray'
                label = f'0/{dfg_count}'
                penwidth = '1'

            dot.edge(src, tgt, label=label, color=color, penwidth=penwidth)

        # Add legend
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Legend', style='dashed')
            legend.node('high_acc', 'High Accuracy', fillcolor=self.get_color_for_accuracy(0.9))
            legend.node('mid_acc', 'Medium Accuracy', fillcolor=self.get_color_for_accuracy(0.5))
            legend.node('low_acc', 'Low Accuracy', fillcolor=self.get_color_for_accuracy(0.1))
            legend.edge('high_acc', 'mid_acc', style='invis')
            legend.edge('mid_acc', 'low_acc', style='invis')

        if filename:
            dot.render(filename, view=False, cleanup=True)

        return dot

    def get_transition_details(
        self,
        source: str,
        target: str
    ) -> Optional[TransitionPerformance]:
        """
        Get detailed performance for a specific transition.

        Args:
            source: Source activity
            target: Target activity

        Returns:
            TransitionPerformance object or None if not found
        """
        return self.transition_performance.get((source, target))

    def get_feature_breakdown(
        self,
        source: str,
        target: str,
        feature_name: str
    ) -> pd.DataFrame:
        """
        Get performance breakdown by feature for a specific transition.

        Args:
            source: Source activity
            target: Target activity
            feature_name: Feature to break down by (e.g., 'Resource')

        Returns:
            DataFrame with columns: feature_value, correct, incorrect, total, accuracy
        """
        perf = self.get_transition_details(source, target)
        if perf is None:
            return pd.DataFrame()

        breakdown = perf.get_feature_breakdown(feature_name)

        rows = []
        for value, counts in breakdown.items():
            total = counts['correct'] + counts['incorrect']
            rows.append({
                feature_name: value,
                'correct': counts['correct'],
                'incorrect': counts['incorrect'],
                'total': total,
                'accuracy': counts['correct'] / total if total > 0 else 0
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('total', ascending=False)

        return df

    def get_all_transitions_summary(self) -> pd.DataFrame:
        """
        Get a summary of all transitions with their performance.

        Returns:
            DataFrame with columns: source, target, correct, incorrect, total, accuracy
        """
        rows = []
        for (src, tgt), perf in self.transition_performance.items():
            rows.append({
                'source': src,
                'target': tgt,
                'correct': perf.correct_count,
                'incorrect': perf.incorrect_count,
                'total': perf.total_count,
                'accuracy': perf.accuracy
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('total', ascending=False)

        return df

    def get_worst_transitions(self, n: int = 10, min_count: int = 5) -> pd.DataFrame:
        """
        Get the transitions with lowest accuracy.

        Args:
            n: Number of transitions to return
            min_count: Minimum total count to include

        Returns:
            DataFrame of worst performing transitions
        """
        df = self.get_all_transitions_summary()
        df = df[df['total'] >= min_count]
        return df.nsmallest(n, 'accuracy')

    def get_best_transitions(self, n: int = 10, min_count: int = 5) -> pd.DataFrame:
        """
        Get the transitions with highest accuracy.

        Args:
            n: Number of transitions to return
            min_count: Minimum total count to include

        Returns:
            DataFrame of best performing transitions
        """
        df = self.get_all_transitions_summary()
        df = df[df['total'] >= min_count]
        return df.nlargest(n, 'accuracy')

    def create_interactive_explorer(
        self,
        activity_name_map: Optional[Dict[str, str]] = None
    ) -> 'InteractiveExplorer':
        """
        Create an interactive explorer widget for Jupyter notebooks.

        Args:
            activity_name_map: Optional dict mapping activity IDs to display names

        Returns:
            InteractiveExplorer widget
        """
        return InteractiveExplorer(self, activity_name_map=activity_name_map)


class InteractiveExplorer:
    """
    Interactive widget for exploring transition performance in Jupyter notebooks.
    """

    def __init__(
        self,
        ppm: ProcessPerformanceMap,
        activity_name_map: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the interactive explorer.

        Args:
            ppm: ProcessPerformanceMap instance
            activity_name_map: Optional dict mapping activity IDs to display names
        """
        self.ppm = ppm
        self.activity_name_map = activity_name_map or {}
        self._setup_widgets()

    def _get_display_name(self, activity_id: str) -> str:
        """Get display name for an activity ID."""
        if self.activity_name_map and activity_id in self.activity_name_map:
            return self.activity_name_map[activity_id]
        return activity_id

    def _setup_widgets(self):
        """Set up ipywidgets for interactivity."""
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML, clear_output
            self.widgets = widgets
            self.display = display
            self.HTML = HTML
            self.clear_output = clear_output
        except ImportError:
            raise ImportError("ipywidgets required. Install with: pip install ipywidgets")

        # Get unique activities
        activities = sorted(set(
            [src for src, _ in self.ppm.transition_performance.keys()] +
            [tgt for _, tgt in self.ppm.transition_performance.keys()]
        ))

        # Create options with display names but internal values as IDs
        # Format: list of (display_name, internal_value) tuples
        activity_options = [('(all)', '(all)')]
        for act_id in activities:
            display_name = self._get_display_name(act_id)
            activity_options.append((display_name, act_id))

        # Get categorical features for breakdown
        cat_features = list(self.ppm.categorical_features.keys())
        cat_features.append('prefix_length')

        # Create widgets
        self.source_dropdown = widgets.Dropdown(
            options=activity_options,
            value='(all)',
            description='Source:',
            style={'description_width': 'initial'}
        )

        self.target_dropdown = widgets.Dropdown(
            options=activity_options,
            value='(all)',
            description='Target:',
            style={'description_width': 'initial'}
        )

        self.feature_dropdown = widgets.Dropdown(
            options=cat_features,
            value=cat_features[1] if len(cat_features) > 1 else cat_features[0],
            description='Breakdown by:',
            style={'description_width': 'initial'}
        )

        self.output = widgets.Output()

        # Link widgets to update function
        self.source_dropdown.observe(self._on_change, names='value')
        self.target_dropdown.observe(self._on_change, names='value')
        self.feature_dropdown.observe(self._on_change, names='value')

    def _on_change(self, change):
        """Handle widget value changes."""
        self._update_display()

    def _add_display_names_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add display name columns to a transitions DataFrame."""
        if self.activity_name_map and len(df) > 0:
            df = df.copy()
            df['source_name'] = df['source'].apply(self._get_display_name)
            df['target_name'] = df['target'].apply(self._get_display_name)
            # Reorder columns to show names first
            cols = ['source_name', 'target_name'] + [c for c in df.columns if c not in ['source_name', 'target_name', 'source', 'target']]
            return df[cols]
        return df

    def _update_display(self):
        """Update the output display based on current selections."""
        with self.output:
            self.clear_output(wait=True)

            source = self.source_dropdown.value
            target = self.target_dropdown.value
            feature = self.feature_dropdown.value

            # Get display names for printing
            source_display = self._get_display_name(source) if source != '(all)' else '(all)'
            target_display = self._get_display_name(target) if target != '(all)' else '(all)'

            if source == '(all)' and target == '(all)':
                # Show overall summary
                df = self.ppm.get_all_transitions_summary()
                df = self._add_display_names_to_df(df)
                print(f"All Transitions Summary ({len(df)} transitions)")
                print("=" * 60)
                self.display(df.head(20))

                # Show overall stats
                total_correct = df['correct'].sum()
                total = df['total'].sum()
                print(f"\nOverall Accuracy: {total_correct}/{total} ({total_correct/total:.1%})")

            elif source != '(all)' and target != '(all)':
                # Show specific transition with feature breakdown
                perf = self.ppm.get_transition_details(source, target)
                if perf is None:
                    print(f"No predictions found for transition: {source_display} -> {target_display}")
                    return

                print(f"Transition: {source_display} -> {target_display}")
                print("=" * 60)
                print(f"Correct: {perf.correct_count}")
                print(f"Incorrect: {perf.incorrect_count}")
                print(f"Total: {perf.total_count}")
                print(f"Accuracy: {perf.accuracy:.1%}")

                print(f"\nBreakdown by {feature}:")
                print("-" * 40)
                breakdown_df = self.ppm.get_feature_breakdown(source, target, feature)
                if len(breakdown_df) > 0:
                    self.display(breakdown_df)
                else:
                    print("No data available for this feature breakdown")

                # Show incorrect predictions details
                if perf.incorrect_count > 0:
                    print(f"\nIncorrect Predictions ({perf.incorrect_count}):")
                    print("-" * 40)
                    incorrect = [p for p in perf.predictions if not p.correct]
                    incorrect_df = pd.DataFrame([
                        {
                            'predicted': self._get_display_name(p.predicted_next_activity),
                            'confidence': f'{p.confidence:.2%}',
                            feature: p.features.get(feature, 'N/A'),
                            'case': p.case_id,
                            'prefix_len': p.prefix_length
                        }
                        for p in incorrect  # Show all
                    ])
                    self.display(incorrect_df)

            else:
                # Show filtered transitions
                df = self.ppm.get_all_transitions_summary()
                if source != '(all)':
                    df = df[df['source'] == source]
                if target != '(all)':
                    df = df[df['target'] == target]

                df = self._add_display_names_to_df(df)

                filter_desc = []
                if source != '(all)':
                    filter_desc.append(f"from '{source_display}'")
                if target != '(all)':
                    filter_desc.append(f"to '{target_display}'")

                print(f"Transitions {' '.join(filter_desc)} ({len(df)} transitions)")
                print("=" * 60)
                self.display(df)

    def show(self):
        """Display the interactive explorer."""
        # Initial display
        self._update_display()

        # Create layout
        controls = self.widgets.HBox([
            self.source_dropdown,
            self.target_dropdown,
            self.feature_dropdown
        ])

        self.display(self.widgets.VBox([controls, self.output]))
