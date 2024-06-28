from dataclasses import dataclass, field


@dataclass
class Metric:
    values: list[float] = field(default_factory=list)
    log_average : bool = False
    
    def add(self, new_values : list[float]):
        self.values += new_values

    @property
    def value(self):
        sum_value = sum(self.values)

        if self.log_average:
            value = sum_value / len(self.values)
        else:
            value = sum_value
        return value
