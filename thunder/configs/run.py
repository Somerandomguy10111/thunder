from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from dataclasses import field
from typing import Iterable, Tuple, Optional

import torch
import wandb
from torch import Tensor
from torch.optim import Optimizer

from thunder.logging.loggers import WBLogger

# ---------------------------------------------------------

@dataclass
class Descent(ABC):
    lr : Optional[float] = None

    def __post_init__(self):
        self.algorithm : type[Optimizer]  = self.get_algorithm()

    def get_optimizer(self, params: Iterable[Tensor]) -> Optimizer:
        kwargs = {k: v for k, v in asdict(self).items() if v is not None}
        return self.algorithm(params, **kwargs)

    @abstractmethod
    def get_algorithm(self) -> type[Optimizer]:
        pass


@dataclass
class Adam(Descent):
    betas: Optional[Tuple[float, float]] = None
    weight_decay: Optional[float] = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.Adam

@dataclass
class SGD(Descent):
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.SGD

@dataclass
class Adagrad(Descent):
    lr_decay: Optional[float] = None
    weight_decay: Optional[float] = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.Adagrad

@dataclass
class Adadelta(Descent):
    weight_decay: Optional[float] = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.Adadelta

@dataclass
class RunConfigs:
    epochs : int = 1
    batch_size : int = 32
    descent: Descent = field(default_factory=Adam)
    save_folderpath = os.path.expanduser(f'~/.py_thunder')
    save_on_done : bool = True
    save_on_epoch : bool = True
    project_name : str = 'unnamed_project'
    run_name : Optional[str] = None
    enable_logging : bool = False


    def make_wandb_logger(self) -> WBLogger:
        config = {
            'lr': self.descent.lr,
            'batch_size': self.batch_size,
            'optimizer': self.descent.get_algorithm().__name__,
            'epochs': self.epochs,
            'model_architecture': 'unnamed architecture',
            'dataset': 'unnamed dataset',
            'experiment_name': 'unnamed experiment',
            'step_metric' : 'epoch',
        }
        log_dirpath = os.path.expanduser(path='~/.wb_logs')
        wandb_run = wandb.init(project=self.project_name, name=self.run_name, config=config, dir=log_dirpath)
        return WBLogger(run=wandb_run)


