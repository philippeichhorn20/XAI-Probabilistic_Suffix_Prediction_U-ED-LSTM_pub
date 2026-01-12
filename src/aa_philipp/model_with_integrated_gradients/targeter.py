from abc import ABC, abstractmethod

class Targeter(ABC):

    @abstractmethod
    def target_fn(self, prediction):
        pass


class ActivityTargeter(Targeter):
    def __init__(self, activity_index):
        self.activity_index = activity_index

    def target_fn(self, preds):
        return preds[0][0]["Activity_mean"][0][self.activity_index].unsqueeze(0)
    

class TimeTargeter(Targeter):
    def target_fn(self, preds):
        return preds[0][1]["case_elapsed_time_mean"][0][0].unsqueeze(0)