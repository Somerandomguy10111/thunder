from dataclasses import dataclass


@dataclass
class Metric:
    num_calls : int = 0
    sum_value : float = 0
    log_average : bool = False
    
    def increment(self, value : float, num_calls : int = 1):
        self.num_calls += num_calls
        self.sum_value += value


    @property
    def value(self):
        if self.log_average:
            value = self.sum_value / self.num_calls
        else:
            value = self.sum_value
        return value
