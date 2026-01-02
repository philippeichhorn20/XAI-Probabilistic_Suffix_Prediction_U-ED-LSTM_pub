import math
from abc import ABC, abstractmethod
import datetime


class Event(ABC):
    # holds a single event which forms the bases of suffixes. for each dataset we implement a child class method
    @abstractmethod
    def from_dict_row(cls, row:dict):
        pass

class HelpdeskEvent(Event):
    # Canonical constructor
    def __init__(self, activity: str, resources: str, case_time, event_time):
        self.activity = activity
        self.resources = resources

        # in seconds
        if isinstance(case_time, (int, float)):
            self.case_elapsed_time = case_time
        else:
            self.case_elapsed_time = case_time[0]
        
        # in seconds
        if isinstance(event_time, (int, float)):
            self.event_elapsed_time = event_time
        else:
            self.event_elapsed_time = event_time[0]


    def __repr__(self):
        pretty_time = str(datetime.timedelta(seconds=self.event_elapsed_time))
        return f"{self.activity} - {pretty_time}"

    @classmethod
    def from_dict_row(cls, row: dict):
        return cls(
            activity=row.get('Activity'),
            resources=row.get('Resource'),
            case_time=row.get('case_elapsed_time', 0.0),
            event_time=row.get('event_elapsed_time', 0.0)
        )
