from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from .configs import ThunderConfig


@dataclass
class Descent(ThunderConfig, ABC):
    lr : float = None

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
    betas: Tuple[float, float] = None
    weight_decay: float = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.Adam


@dataclass
class SGD(Descent):
    momentum: float = None
    weight_decay: float = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.SGD


@dataclass
class Adagrad(Descent):
    lr_decay: float = None
    weight_decay: float = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.Adagrad


@dataclass
class Adadelta(Descent):
    weight_decay: float = None

    def get_algorithm(self) -> type[Optimizer]:
        return torch.optim.Adadelta
