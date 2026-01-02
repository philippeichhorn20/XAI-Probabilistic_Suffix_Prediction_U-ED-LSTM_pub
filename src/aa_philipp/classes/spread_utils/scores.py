from classes.sequence import Sequence

import networkx as nx


"""
Idea: Split the scoring into three component:
1. Accuracy of the predicted length
2. Correctness of the activity sequence using levenshtein distance
3. Accuracy of the timestamp prediction


"""


def get_weighted_dist(seq_a: Sequence, 
                      seq_b: Sequence, 
                      activity_weight:int = 1, 
                      length_weight:int = 1, 
                      temp_weight:int = 1):
    
    activity_dist = get_activity_dist(seq_a, seq_b)*activity_weight
    length_dist = get_length_dist(seq_a, seq_b)*length_weight
    temp_dist_in_days = get_temp_dist_in_days(seq_a, seq_b)*temp_weight
    return activity_dist, length_dist, temp_dist_in_days


def get_activity_dist(seq_a: Sequence, seq_b: Sequence) -> int:
    graph_a: nx.DiGraph = Sequence.get_as_nx_graph(seq_a)
    graph_b: nx.DiGraph = Sequence.get_as_nx_graph(seq_b)

    edit_distance = nx.graph_edit_distance(graph_a, graph_b)

    return edit_distance

def get_length_dist(seq_a: Sequence, seq_b: Sequence) -> int:
    difference = len(seq_a.event_list)-len(seq_b.event_list)
    abs_difference = abs(difference)
    return abs_difference


def get_temp_dist_in_days(seq_a: Sequence, seq_b: Sequence) -> float:
    considered_events_num = min(len(seq_a.event_list), len(seq_b.event_list))
    if considered_events_num == 0:
        
        return 0.0

    acc = 0.0
    for i in range(considered_events_num):
        # convert seconds to days by dividing by 86400
        days_a = seq_a.event_list[i].event_elapsed_time / 86400.0
        days_b = seq_b.event_list[i].event_elapsed_time / 86400.0
        abs_difference_days = abs(days_a - days_b)
        acc += abs_difference_days
    result = acc / considered_events_num
    return result

