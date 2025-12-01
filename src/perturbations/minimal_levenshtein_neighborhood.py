from perturbations.event import Event
from rapidfuzz.distance import Levenshtein

from perturbations.trace import Trace
from perturbations.perturbations import Perturbations

class Minimal_Levenshtein_Neighborhood(Perturbations):
    """
    This method samples perturbations from the list_of_traces and
    returns n-traces that have the smallest levenstein distance from the original
    """

    def __init__(self, list_of_traces:list[Trace]):
        self.list_of_traces = list_of_traces # this is where the perturbations is sampled from 

    def generate(self, trace:Trace, number_of_perturbations:int, prefix_length:int)->list[Event]:
        # Use only the prefix activity sequence for comparison
        original_activities = [event.category for event in trace.event_list[:prefix_length]]

        distance_trace_pairs = []
        for candidate_trace in self.list_of_traces:
            candidate_activities = [event.category for event in candidate_trace.event_list[:prefix_length]]
            dist = Levenshtein.distance(tuple(original_activities), tuple(candidate_activities))
            distance_trace_pairs.append((dist, candidate_trace))

        # sort by distance (ascending) and select up to 'number_of_perturbations' traces
        distance_trace_pairs.sort(key=lambda x: x[0])
        closest_traces = [pair[1] for pair in distance_trace_pairs[:number_of_perturbations]]

        return closest_traces # the closest trace should be itself
