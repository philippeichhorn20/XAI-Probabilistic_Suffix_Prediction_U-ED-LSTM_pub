from collections import Counter
from typing import List, Dict, Iterable, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx


Event = Dict[str, object]
Trace = List[Event]
Traces = List[Trace]
Edge = Tuple[str, str]


class SuffixDFGVisualizer:
    """
    Build a probabilistic Directly-Follows Graph (DFG) from predicted suffixes
    and overlay the correct (actual) suffix path.

    Typical usage:
        viz = SuffixDFGVisualizer(ignore_activities={"EOS"})
        viz.fit_predictions(predicted_suffixes)
        viz.set_correct_suffix(correct_suffix)
        viz.plot(min_prob=0.05)
    """

    def __init__(self, ignore_activities: Optional[Iterable[str]] = None):
        self.ignore_activities: Set[str] = set(ignore_activities or [])
        self.edge_counts: Counter[Edge] = Counter()
        self.edge_probs: Dict[Edge, float] = {}
        self.n_traces: int = 0
        self.actual_edges: Set[Edge] = set()
        self.G: Optional[nx.DiGraph] = None


    def _suffix_to_edges(self, trace: Trace) -> List[Edge]:
        """Convert a trace to directly-follows edges using only 'Activity'."""
        filtered = [e for e in trace if e.get("Activity") not in self.ignore_activities]
        return [
            (filtered[i]["Activity"], filtered[i + 1]["Activity"])
            for i in range(len(filtered) - 1)
        ]


    def fit_predictions(self, predicted_suffixes: Traces) -> None:
        """
        Ingest predicted suffixes and compute edge frequencies & probabilities.

        :param predicted_suffixes: list of traces (each trace is a list of events)
        """
        self.edge_counts.clear()
        self.edge_probs.clear()
        self.n_traces = 0

        for trace in predicted_suffixes:
            if not trace:
                continue
            edges = self._suffix_to_edges(trace)
            if not edges:
                continue
            self.edge_counts.update(edges)
            self.n_traces += 1

        if self.n_traces == 0:
            raise ValueError("No non-empty traces with edges were provided.")

        self.edge_probs = {e: c / self.n_traces for e, c in self.edge_counts.items()}
        self.G = None  # will be rebuilt in _build_graph

    def set_correct_suffix(self, correct_suffix: Trace) -> None:
        """
        Set the actual (correct) suffix path for the current case.

        :param correct_suffix: trace (list of events) representing the true future.
        """
        self.actual_edges = set(self._suffix_to_edges(correct_suffix))

    # ---------- Graph building ----------

    def _build_graph(self, min_prob: float = 0.0) -> None:
        """
        Build the networkx DiGraph from edge probabilities and actual edges.
        """
        if not self.edge_probs:
            raise RuntimeError("fit_predictions() must be called before _build_graph().")

        G = nx.DiGraph()

        # Add predicted edges above min_prob threshold
        for (u, v), p in self.edge_probs.items():
            if p >= min_prob:
                G.add_edge(u, v, prob=p)

        # Ensure nodes from actual suffix are present even if they have low prob
        for (u, v) in self.actual_edges:
            if not G.has_node(u):
                G.add_node(u)
            if not G.has_node(v):
                G.add_node(v)

        self.G = G


    def plot(
        self,
        min_prob: float = 0.0,
        layout: str = "spring",
        figsize: Tuple[int, int] = (8, 6),
        show_edge_probs: bool = True,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the probabilistic DFG with the actual path highlighted.

        :param min_prob: minimum probability for edges to be shown
        :param layout: 'spring', 'kamada_kawai', or 'circular'
        :param figsize: size of the matplotlib figure
        :param show_edge_probs: whether to draw edge labels with probabilities
        :param title: optional title for the plot
        :return: matplotlib Figure
        """
        self._build_graph(min_prob=min_prob)
        assert self.G is not None, "Graph not built."

        G = self.G

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        fig, ax = plt.subplots(figsize=figsize)

        # Nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=1500,
            node_color="white",
            edgecolors="black",
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # Base predicted edges (thickness proportional to probability)
        base_edges = list(G.edges())
        base_widths = [1 + 6 * G[u][v]["prob"] for u, v in base_edges]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=base_edges,
            width=base_widths,
            edge_color="lightgray",
            arrowsize=30,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

        # Actual path edges overlaid
        actual_edges_list = [e for e in self.actual_edges if e in G.edges()]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=actual_edges_list,
            width=3,
            edge_color="red",
            arrowsize=20,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

        # Edge probability labels
        if show_edge_probs:
            edge_labels = {
                (u, v): f"{G[u][v]['prob']:.2f}" for u, v in G.edges()
            }
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=7,
                ax=ax,
            )

        ax.set_axis_off()
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    @classmethod
    def visualize(
        cls,
        predicted_suffixes: Traces,
        correct_suffix: Trace,
        ignore_activities: Optional[Iterable[str]] = None,
        **plot_kwargs,
    ) -> plt.Figure:
        """
        One-shot helper: build visualizer, fit predictions, set correct suffix, and plot.

        Example:
            SuffixDFGVisualizer.visualize(
                predicted_suffixes,
                correct_suffix,
                ignore_activities={"EOS"},
                min_prob=0.05,
                layout="spring",
                title="Case 123 – prediction vs actual"
            )
        """
        viz = cls(ignore_activities=ignore_activities)
        viz.fit_predictions(predicted_suffixes)
        viz.set_correct_suffix(correct_suffix)
        return viz.plot(**plot_kwargs)
