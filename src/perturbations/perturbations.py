from abc import ABC, abstractmethod

from perturbations.trace import Trace
from perturbations.event import Event

class Perturbations(ABC):

    @abstractmethod
    def generate(trace:Trace)->list[Event]:
        pass