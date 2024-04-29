from __future__ import annotations

from abc import abstractmethod
from typing import Optional
from torch import Tensor

class Viewer:
    def __init__(self):
        self.sample : Optional[Tensor] = None

    @abstractmethod
    def view(self, batch : Tensor, output : Tensor):
        pass

    def save(self, output : Tensor):
        pass

    # def get_sample_batch(self) -> Optional[Tensor]:
    #     return self.sample_batch
    #
    # def set_sample_batch(self, batch : Tensor):
    #     self.sample_batch = batch
