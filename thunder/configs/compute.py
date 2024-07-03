from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import torch
from torch import dtype, device


# ---------------------------------------------------------

@dataclass
class ComputeConfigs:
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dtype : dtype = torch.float32
    allow_tensor_cores : bool = True

    def get_num_devices(self) -> int:
        num_devices = self.num_gpus if self.num_gpus > 0 else os.cpu_count()//2
        return num_devices

    @property
    def device(self) -> torch.device:
        torch_device = device('cuda') if self.num_gpus > 0 else torch.device('cpu')
        return torch_device

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