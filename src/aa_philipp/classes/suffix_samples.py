import math
import collections
import networkx as nx
import numpy as np

from classes.sequence import Sequence
from classes.event import Event


import classes.spread_utils.scores as Scores

class SuffixSamples:
    """
    A container class holding a list of Suffix objects,
    representing the entire dataset or trace log.
    """

    def __init__(
        self,
        predicted_suffixes: list[Sequence],
        case_name: str,
        prefix: Sequence,
        suffix: Sequence,
        mean_prediction: Sequence,
    ):
        self.predicted_suffixes = predicted_suffixes
        self.case_name = case_name
        self.prefix = prefix
        self.suffix = suffix
        self.mean_prediction = mean_prediction

    def __len__(self):
        return len(self.predicted_suffixes)

    def __repr__(self):
        return (
            f"SuffixSamples(case_name={self.case_name}, "
            f"prefix={self.prefix}, "
            f"suffix={self.suffix}, "
            f"mean_prediction={self.mean_prediction}, "
            f"suffix_len={len(self.suffix.event_list)}, "
            f"num_predicted_suffixes={len(self.predicted_suffixes)}"
            f")"
        )

    @classmethod
    def from_row(cls, row):
        """
        Processes the entire list of lists (raw_data) into SuffixSamples.
        """
        predicted_suffixes = [
            Sequence.from_raw_group(group) for group in row["predicted_suffixes"]
        ]
        case_name = row["case_name"]
        prefix = Sequence.from_raw_group(row["prefix"])
        suffix = Sequence.from_raw_group(row["suffix"])
        mean_prediction = Sequence.from_raw_group(row["mean_prediction"])

        return cls(predicted_suffixes, case_name, prefix, suffix, mean_prediction)

    @staticmethod
    def calculate_shannon_entropy_on_activity_suffix(row, base: int = 2) -> float:
        """
        Calculates the Shannon Entropy based on the distribution of
        unique entire suffix sequences.

        Just fyi: high entropy <-> high uncertainty
        """

        suffix_samples = SuffixSamples.from_row(row=row)

        suffix_sequence_counts = {}
        total_suffixes = 0

        for suffix in suffix_samples.predicted_suffixes:

            sequence_key = tuple(event.activity for event in suffix.event_list)

            suffix_sequence_counts[sequence_key] = (
                suffix_sequence_counts.get(sequence_key, 0) + 1
            )
            total_suffixes += 1

        total_outcomes = total_suffixes

        if total_outcomes == 0:
            return 0.0

        shannon_entropy = 0.0

        for count in suffix_sequence_counts.values():
            probability = count / total_outcomes

            if probability > 0:
                log_p = math.log(probability) / math.log(base)
                shannon_entropy += probability * log_p

        return -shannon_entropy

    @staticmethod
    def calculate_brier_score_full_suffix(row: dict) -> float:

        """
        Calculates the brier score on the full suffix (activities only)


        0.0: all predictions match the true sequence
        1.0: no predictions match the true sequence
        """

        suffix_samples = SuffixSamples.from_row(row=row)

        true_activity_sequence = tuple(
            event.activity for event in suffix_samples.suffix.event_list
        )

        sequence_counts = collections.defaultdict(int)
        total_predictions = 0

        for predicted_seq in suffix_samples.predicted_suffixes:
            predicted_activity_sequence = tuple(
                event.activity for event in predicted_seq.event_list
            )

            comparison_sequence = predicted_activity_sequence[
                : len(true_activity_sequence)
            ]

            if len(predicted_activity_sequence) >= len(true_activity_sequence):
                sequence_counts[comparison_sequence] += 1

            total_predictions += 1

        if total_predictions == 0:
            return float("nan")

        count_true_sequence = sequence_counts.get(true_activity_sequence, 0)

        predicted_probability = count_true_sequence / total_predictions

        brier_score = (predicted_probability - 1.0) ** 2

        return brier_score

    def get_avg_leventshtein_dist_to_actual_suffix(self):
        """
        This function caluclate the leventstein distance (edit distance) between all the sampled
        predictions and the actual suffix.
        """
        edit_dists: list = SuffixSamples.get_list_of_edit_distances(
            self.predicted_suffixes, self.suffix
        )
        avg = sum(edit_dists) / len(edit_dists)
        return avg

    def get_avg_leventshtein_dist_to_predicted_suffix(self):
        """
        This function caluclate the leventstein distance (edit distance) between all the sampled
        predictions and the actual suffix.
        """
        edit_dists: list = SuffixSamples.get_list_of_edit_distances(
            self.predicted_suffixes, self.mean_prediction
        )
        avg = sum(edit_dists) / len(edit_dists)
        return avg

    @staticmethod
    def get_list_of_edit_distances(
        set_of_sequences: list[Sequence], benchmark_sequence: Sequence
    ):

        benchmark_graph = benchmark_sequence.get_as_nx_graph()

        set_of_sequence_graph_list = []

        for seq in set_of_sequences:
            set_of_sequence_graph_list.append(seq.get_as_nx_graph())

        list_of_distances = []

        for sampled_graph in set_of_sequence_graph_list:
            lev_dist = nx.graph_edit_distance(benchmark_graph, sampled_graph)
            list_of_distances.append(lev_dist)
        return list_of_distances
    
    @staticmethod
    def get_scoring_measure(row):

        suffix_samples = SuffixSamples.from_row(row)

        scores = []
        for predicted_suffix in suffix_samples.predicted_suffixes:
            activity_dist, length_dist, temp_dist_in_days = Scores.get_weighted_dist(
                predicted_suffix, suffix_samples.suffix
            )
            distances = (activity_dist, length_dist, temp_dist_in_days)
            scores.append(distances)

        if scores:
            scores_tensor = np.array(scores)
            return np.mean(scores_tensor, axis=0)
        else:
            
            return np.array([0.0, 0.0, 0.0])

