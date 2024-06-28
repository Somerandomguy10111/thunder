from dataclasses import dataclass


@dataclass
class Metric:
    name: str
    num_calls : int = 0
    sum_value : float = 0
    log_average : bool = False
    
    def increment(self, value : float):
        self.num_calls += 1
        self.sum_value += value


    @property
    def value(self):
        if self.log_average:
            value = self.sum_value / self.num_calls
        else:
            value = self.sum_value
        return value
