import math
import collections
import networkx as nx

from classes.event import Event, HelpdeskEvent

class Sequence:
    def __init__(self, event_list: list[Event]):
        self.event_list = event_list

    def __repr__(self):
        repr_str = f"Sequence(len={len(self.event_list)}, events=[\n"
        for event in self.event_list:
            repr_str += f"  {event}\n"
        repr_str += "])"
        return repr_str

    @classmethod 
    def from_raw_group(cls, raw_group_list: list):
        if not raw_group_list:
            return cls([])
            
        events = [HelpdeskEvent.from_dict_row(row) for row in raw_group_list]
        return cls(events)
    

    def get_as_nx_graph(self):
        G = nx.DiGraph()
        prev_node = None
        for event in self.event_list:
            if event.activity is None:
                event.activity = "None"

            G.add_node(event.activity)
            if prev_node is not None:
                G.add_edge(prev_node.activity,event.activity)
            prev_node = event
        return G
