from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import dtype, device
from torch.utils.data import Dataset

# ---------------------------------------------------------

@dataclass
class ComputeConfigs:
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dtype : dtype = torch.float32

    def get_num_devices(self) -> int:
        num_devices = self.num_gpus if self.num_gpus > 0 else os.cpu_count()//2
        return num_devices

    @property
    def device(self) -> torch.device:
        torch_device = device('cuda') if self.num_gpus > 0 else torch.device('cpu')
        return torch_device


class ThunderDataset(Dataset):
    def __init__(self, dataset : Dataset, torch_device : device, torch_dtype : dtype):
        self.base_dataset : Dataset = dataset
        self.device : device = torch_device
        self.dtype : dtype = torch_dtype

    # noinspection PyTypeChecker
    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, idx):
        content = self.base_dataset[idx]
        if isinstance(content, tuple):
            data, label = content
            if isinstance(data, torch.Tensor):
                data = data.to(dtype=self.dtype, device=self.device)
            if isinstance(label, torch.Tensor):
                label = label.to(dtype=self.dtype, device=self.device)
            return data, label
        elif isinstance(content, torch.Tensor):
            content = content.to(dtype=self.dtype, device=self.device)
        else:
            content = content

        return content
