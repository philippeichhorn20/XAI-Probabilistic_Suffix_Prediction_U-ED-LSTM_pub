import csv
from datetime import datetime
from perturbations.event import Event


class Trace:
    def __init__(self, event_list: list, name: str):
        self.event_list = event_list
        self.name = name

    @staticmethod
    def from_csv(csv_path:str, trace_name_column:str, activity_column:str, timestamp_column:str):
        """
        Create a list of Traces from a CSV file, one Trace per unique trace_name_column value.
        """
        traces_dict = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace_name = row[trace_name_column]
                activity = row[activity_column]
                timevalue = row[timestamp_column]
                if trace_name not in traces_dict:
                    traces_dict[trace_name] = Trace(event_list=[], name=trace_name)
                traces_dict[trace_name].event_list.append(
                    Event(category=activity, timevalue=timevalue, parent_trace=traces_dict[trace_name])
                )
        # Ensure sorted by time
        for trace in traces_dict.values():
            trace.event_list.sort(key=lambda e: e.timevalue)
        return list(traces_dict.values())

    def __str__(self):
        event_strs = ["\n\t" + str(event) for event in self.event_list]
        return f"Trace(name={self.name}, events=[{', '.join(event_strs)}])"

    def to_csv(self, csv_path, trace_name_column="Case ID", activity_column="Activity", timestamp_column="Complete Timestamp"):
        """
        Export this Trace object to a CSV file.
        - csv_path: output CSV file path
        - trace_name_column, activity_column, timestamp_column: output column names
        """
        rows = []
        for event in self.event_list:
            # Assume event.timevalue is already string or is formatted as needed
            rows.append({
                trace_name_column: self.name,
                activity_column: event.category,
                timestamp_column: event.timevalue
            })
        # Write to CSV
        if rows:
            fieldnames = [trace_name_column, activity_column, timestamp_column]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def deepcopy(self):
        """Creates a deep copy of this Trace, with all events' parent_trace set to the new Trace."""
        # Create a new empty Trace object
        new_trace = Trace(event_list=[], name=self.name)
        # Copy the events, setting parent_trace to new_trace for each
        new_events = []
        for event in self.event_list:
            # Re-create the event with the same attributes but set new parent_trace
            copied_event = Event(category=event.category, timevalue=event.timevalue, parent_trace=new_trace)
            new_events.append(copied_event)
        new_trace.event_list = new_events
        return new_trace
    
     
    def remove_event_at(self, index:int):
        self.event_list.pop(index)
