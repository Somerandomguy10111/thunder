from __future__ import annotations
import torch
from typing import Iterator

class Tensor1D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> Tensor1D:
        if tensor.dim() != 1:
            raise ValueError('Tensor1D must have 1 dimension')
        return tensor.as_subclass(cls)

    def __iter__(self) -> Iterator[torch.Tensor]:
        raise ValueError('Cannot iterate over a Tensor1D')

    def __getitem__(self, item) -> torch.Tensor:
        item = super().__getitem__(item)
        return item  # Returns a scalar (0-dimensional tensor).

class Tensor2D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> Tensor2D:
        if tensor.dim() != 2:
            raise ValueError('Tensor2D must have 2 dimensions')
        return tensor.as_subclass(cls)

    def __iter__(self) -> Iterator[Tensor1D]:
        for t in super().__iter__():
            yield Tensor1D(t)

    def __getitem__(self, item) -> Tensor1D:
        item = super().__getitem__(item)
        return Tensor1D(item)

class Tensor3D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> Tensor3D:
        if tensor.dim() != 3:
            raise ValueError('Tensor3D must have 3 dimensions')
        return tensor.as_subclass(cls)

    def __iter__(self) -> Iterator[Tensor2D]:
        for t in super().__iter__():
            yield Tensor2D(t)

    def __getitem__(self, item) -> Tensor2D:
        item = super().__getitem__(item)
        return Tensor2D(item)

class Tensor4D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor) -> Tensor4D:
        if tensor.dim() != 4:
            raise ValueError('Tensor4D must have 4 dimensions')
        return tensor.as_subclass(cls)

    def __iter__(self) -> Iterator[Tensor3D]:
        for t in super().__iter__():
            yield Tensor3D(t)

    def __getitem__(self, item) -> Tensor3D:
        item = super().__getitem__(item)
        return Tensor3D(item)