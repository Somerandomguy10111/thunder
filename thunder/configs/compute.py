from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import torch

# ---------------------------------------------------------

@dataclass
class ComputeConfig:
    device : torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype : torch.dtype = torch.float32
    allow_tensor_cores : bool = False

    def __post_init__(self):
        if self.device.type == 'cuda' and self.device.index:
            if self.device.index >= torch.cuda.device_count():
                raise ValueError(f'CUDA device \"{self.device}\" does not exist')

    def get_num_gpus(self) -> int:
        if self.device.type == 'cuda':
            value = torch.cuda.device_count() if self.device.index is None else 1
        else:
            value = 0
        return value

    @staticmethod
    def get_num_cpus() -> int:
        return os.cpu_count()

    def __str__(self):
        the_str = f'ComputeConfig:\n'
        for k,v in dataclasses.asdict(self).items():
            the_str += f'-{k}: {v}\n'
        return the_str


if __name__ == "__main__":
    configs = ComputeConfig(device=torch.device(f'cuda:0'))
    print(configs.get_num_gpus())
