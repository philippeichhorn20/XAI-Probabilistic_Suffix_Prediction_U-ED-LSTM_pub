import pandas as pd
from classes.sequence import Sequence

class PrefixCounter:
    dataset: pd.DataFrame

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

        # Precompute traces grouped by Case ID
        # Each value is a list of activities in order
        self.traces = (
            dataset.sort_values("Complete Timestamp")
                   .groupby("Case ID")["Activity"]
                   .apply(list)
        )

    def count_prefix_occurences(self, prefix: Sequence) -> int:
        """
        Count how many traces start with the given prefix.
        """
        # Convert Sequence into a list of activity strings

        prefix_list = [event.activity for event in prefix]


        plen = len(prefix_list)

        count = 0

        for trace in self.traces:
            # Check if start of trace equals the prefix
            if trace[:plen] == prefix_list:
                count += 1

        return count
