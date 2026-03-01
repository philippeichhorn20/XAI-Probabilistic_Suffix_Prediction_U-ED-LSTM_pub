"""Event log utilities via pm4py."""

import pandas as pd
import pm4py


def csv_to_xes(csv_path, xes_path, case_id_col="case_id", activity_col="activity", timestamp_col="timestamp"):
    """Convert CSV to XES."""
    df = pd.read_csv(csv_path)
    df = pm4py.format_dataframe(df, case_id=case_id_col, activity_key=activity_col, timestamp_key=timestamp_col)
    pm4py.write_xes(pm4py.convert_to_event_log(df), str(xes_path))
