from perturbations.perturbations import Perturbations
from perturbations.event import Event
from perturbations.trace import Trace


class LeaveOneOutPerturbations(Perturbations):

    
    def generate(trace:Trace, prefix_length:int=1000000000)->list[Event]:
        '''
        Creates a list of (deep copied) traces that are changed 
        from the original in that one of the prefix activities is removed
        i.e. for [a,b,c,d,e] and prefix_length=3
        it returns -> [[b,c,d,e],[a,c,d,e],[a,b,d,e]]
        '''

        trace_length = len(trace.event_list)
        perturbed_traces = []
        for index in range(min(prefix_length,trace_length)):
            trace_copy = trace.deepcopy()
            trace_copy.remove_event_at(index)
            perturbed_traces.append(trace_copy)
        return perturbed_traces
