from trace import Trace

class Event:
    def __init__(self, category, timevalue, parent_trace:Trace):
        self.category = category
        self.timevalue = timevalue
        self.parent_trace = parent_trace

    def __str__(self):
        return f"Event(category={self.category}, timevalue={self.timevalue})"