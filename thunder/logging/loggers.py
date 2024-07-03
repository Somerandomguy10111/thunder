from __future__ import annotations

import os

from GPUtil import GPU
from wandb.sdk.wandb_run import Run

from holytools.logging import LoggerFactory
from .metric import Metric

thunderLogger = LoggerFactory.make_logger(name=__name__)



# ---------------------------------------------------------
class WBLogger:
    def __init__(self, run : Run):
        self.run : Run = run
        self.current_batch : int = 0
        self.current_epoch : int = 0

    @classmethod
    def wandb_is_available(cls) -> bool:
        if os.getenv('WANDB_API_KEY'):
            return True
        elif os.path.isfile(os.path.expanduser('~/.netrc')):
            return True
        else:
            return False
    # ---------------------------------------------------------
    # increment

    def increment_epoch(self):
        self.current_epoch += 1

    def increment_batch(self):
        self.current_batch += 1

    # ---------------------------------------------------------
    # logging (interface)

    def log_metrics(self, metric_map : dict[str, Metric], is_training: bool):
        for k, v in metric_map.items():
            if is_training:
                self.log_training_quantity(name=k, value=v.value)
            else:
                self.log_validation_quantity(name=k, value=v.value)

    def log_gpu_resources(self, gpus : list[GPU]):
        for gpu in gpus:
            gpu_memory_load_factor = (gpu.memoryTotal - gpu.memoryFree) / gpu.memoryTotal
            self.log_system(name=f'GPU {gpu.id} free memory in GB', value=gpu.memoryFree / 1024)
            self.log_system(name=f'GPU {gpu.id} memory load', value=gpu_memory_load_factor)
        free_gpu_memory_mb = sum([gpu.memoryFree for gpu in gpus])
        total_gpu_memory_mb = sum([gpu.memoryTotal for gpu in gpus])
        memory_load_factor = (total_gpu_memory_mb - free_gpu_memory_mb) / total_gpu_memory_mb
        self.log_system(name='Free GPU memory in GB', value=free_gpu_memory_mb / 1024)
        self.log_system(name='GPU memory load', value=memory_load_factor)

    def log_validation_quantity(self, name: str, value: float):
        self.log_quantity(name=f'Validation/{name}', value=value)

    def log_training_quantity(self, name: str, value: float):
        self.log_quantity(name=f'Training/{name}', value=value)

    def log_system(self, name : str, value : float):
        self.log_quantity(name=f'System/{name}', value=value)

    def log_quantity(self, name: str, value: float):
        self._log(metric_dict={name: value})

    # ---------------------------------------------------------
    # logging (internal)

    def _log(self, metric_dict: dict[str, int | float]):
        metric_dict['epoch'] = self.current_epoch
        metric_dict['batch'] = self.current_batch
        self.run.log(data=metric_dict)




