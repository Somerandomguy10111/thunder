from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import wandb
from torch import device, dtype
from torch.utils.data import Dataset

from .descent import Descent, Adam


# ---------------------------------------------------------

class ThunderConfig:
    pass

@dataclass
class ComputeConfigs(ThunderConfig):
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dtype : dtype = torch.float32

    def get_num_devices(self) -> int:
        num_devices = self.num_gpus if self.num_gpus > 0 else os.cpu_count()//2
        return num_devices

    @property
    def device(self) -> torch.device:
        torch_device = device('cuda') if self.num_gpus > 0 else torch.device('cpu')
        return torch_device


@dataclass
class RunConfigs:
    epochs : int = 1
    batch_size : int = 32
    descent: Descent = field(default_factory=Adam)
    print_full_stacktrace : bool = False
    save_folderpath = os.path.expanduser(f'~/.py_thunder')
    save_on_done : bool = True
    save_on_epoch : bool = True


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

    @classmethod
    def from_runconfigs(cls, run_configs : RunConfigs, project_name : str = 'unnamed_project'):
        return cls(lr=run_configs.descent.lr,
                   batch_size=run_configs.batch_size,
                   optimizer=run_configs.descent.get_algorithm().__name__,
                   epochs=run_configs.epochs,
                   project_name=project_name)


    def get_logger(self):
        kwargs = asdict(self)
        del kwargs['project_name']
        wandb_run = wandb.init(project=self.project_name, config=kwargs)
        return wandb_run


class ComputeConformDataset(Dataset):
    def __init__(self, base_dataset : Dataset, torch_device : device, torch_dtype : dtype):
        self.base_dataset : Dataset = base_dataset
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


if __name__ == "__main__":
