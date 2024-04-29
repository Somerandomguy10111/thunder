from __future__ import annotations

import os
import torch

from dataclasses import asdict
from pytorch_lightning.loggers import Logger, WandbLogger

from dataclasses import dataclass
from torch import device, dtype
from torch.utils.data import Dataset
from typing import Optional
from .descent import Descent, Adam
# ---------------------------------------------------------

class ThunderConfig:
    pass

@dataclass
class ComputeConfigs(ThunderConfig):
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dtype : dtype = torch.float32

    def get_accelerator(self) -> str:
        return "gpu" if self.num_gpus > 0 else "cpu"

    @property
    def device(self) -> torch.device:
        torch_device = device('cuda') if self.num_gpus > 0 else torch.device('cpu')
        return torch_device


@dataclass
class RunConfigs:
    epochs : int = 1
    batch_size : int = 32
    seed : int = 42
    descent: Descent = Adam()
    checkpoint_on_epoch : bool = True
    print_full_stacktrace : bool = False
    save_folderpath = os.path.expanduser(f'~/.py_thunder')



@dataclass
class WBConfig:
    lr: float
    batch_size: int
    optimizer: str
    epochs: int
    model_architecture: str = 'unnamed architecture'
    dataset: str = 'unnamed dataset'
    experiment_name: str = 'unnamed experiment'
    project_name: str = 'unnamed project'
    log_dirpath: str = '~/.wb_logs'
    seed : Optional[int] = None

    def get_logger(self) -> Logger:
        wb_configs = asdict(self)
        return WandbLogger(save_dir=os.path.expanduser(self.log_dirpath), config=wb_configs)



class ComputeConformDataset(Dataset):
    def __init__(self, base_dataset : Dataset, torch_device : device, torch_dtype : dtype):
        self.base_dataset : Dataset = base_dataset
        self.device : device = torch_device
        self.dtype : dtype = torch_dtype

    # noinspection
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
