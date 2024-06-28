from __future__ import annotations

import os

from wandb.sdk.wandb_run import Run

from holytools.logging import LoggerFactory

thunderLogger = LoggerFactory.make_logger(name=__name__)



# ---------------------------------------------------------
class WBLogger:
    def __init__(self, run : Run):
        self.run : Run = run
        self.current_batch : int = 0
        self.current_epoch : int = 0

    # ---------------------------------------------------------
    # increment

    def increment_epoch(self):
        self.current_epoch += 1

    def increment_batch(self):
        self.current_batch += 1

    # ---------------------------------------------------------
    # logging


    def log_validation_quantity(self, name: str, value: float):
        self.log_quantity(name=f'Training/{name}', value=value)

    def log_training_quantity(self, name: str, value: float):
        self.log_quantity(name=f'Validation/{name}', value=value)

    def log_quantity(self, name: str, value: float):
        self.log(metric_dict={name: value})

    def log(self, metric_dict: dict[str, int | float]):
        metric_dict['epoch'] = self.current_epoch
        metric_dict['batch'] = self.current_batch
        self.run.log(data=metric_dict)

    @classmethod
    def wandb_is_available(cls) -> bool:
        if os.getenv('WANDB_API_KEY'):
            return True
        elif os.path.isfile(os.path.expanduser('~/.netrc')):
            return True
        else:
            return False



