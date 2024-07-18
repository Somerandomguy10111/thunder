from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import torch
from torch import dtype, device


# ---------------------------------------------------------

@dataclass
class ComputeConfigs:
    torch_device : device = device('cuda') if torch.cuda.is_available() else device('cpu')
    dtype : dtype = torch.float32
    allow_tensor_cores : bool = False

    def get_num_gpus(self) -> int:
        if self.torch_device.type == 'cuda':
            value = torch.cuda.device_count() if self.torch_device.index is None else 1
        else:
            value = 0
        return value

    @staticmethod
    def get_num_cpus() -> int:
        return os.cpu_count()

    def __str__(self):
        the_str = f'ComputeConfigs:\n'
        for k,v in dataclasses.asdict(self).items():
            the_str += f'-{k}: {v}\n'
        return the_str

class Devices:
    gpu : device = device('cuda')
    cpu : device = device('cpu')


if __name__ == "__main__":
    print(ComputeConfigs())