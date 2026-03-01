"""MP-Declare constraint mining via Declare4Py."""

from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareAnalyzer import MPDeclareAnalyzer


def load_event_log(xes_path):
    """Load XES event log."""
    log = D4PyEventLog(case_name="case:concept:name")
    log.parse_xes_log(str(xes_path))
    return log


def discover_constraints(xes_path, min_support=0.9):
    """Discover MP-Declare constraints from XES log."""
    return DeclareMiner(log=load_event_log(xes_path), min_support=min_support, consider_vacuity=False).run()


def check_conformance(xes_path, model):
    """Check conformance of log against model."""
    return MPDeclareAnalyzer(log=load_event_log(xes_path), declare_model=model, consider_vacuity=False).run()
