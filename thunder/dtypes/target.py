from abc import abstractmethod
from torch import Tensor
from typing import TypeVar

TargetType = TypeVar('TargetType')

# The idea of this class is that you can pass type[Target] as an attribute to the network
# to define its architecture. The interface of the method allos you to
# -> Define the number of outputs dynamically
# -> Have the model predict not just an illegible tensor but return the actual python
# object representation that you would like to work with

class Target:
    @abstractmethod
    def to_tensor(self) -> Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_tensor(cls) -> TargetType:
        pass

    @classmethod
    @abstractmethod
    def get_length(cls):
        pass