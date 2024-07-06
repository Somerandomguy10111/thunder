from __future__ import annotations
import torch
from typing import Iterator

class Tensor1D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> Tensor1D:
        if tensor.dim() != 1:
            raise ValueError('Tensor1D must have 1 dimension')
        return tensor.as_subclass(cls)


class Tensor2D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> Tensor2D:
        if tensor.dim() != 2:
            raise ValueError('Tensor2D must have 2 dimensions')
        return tensor.as_subclass(cls)


class Tensor3D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> Tensor3D:
        if tensor.dim() != 3:
            raise ValueError('Tensor3D must have 3 dimensions')
        return tensor.as_subclass(cls)


class Tensor4D(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor) -> Tensor4D:
        if tensor.dim() != 4:
            raise ValueError('Tensor4D must have 4 dimensions')
        return tensor.as_subclass(cls)