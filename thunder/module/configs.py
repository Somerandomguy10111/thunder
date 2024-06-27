from __future__ import annotations
from dataclasses import dataclass, field
import os
from torch import device, dtype
from torch.utils.data import Dataset
import torch

import wandb
from wandb.wandb_run import Run

from .descent import Descent, Adam

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


@dataclass
class RunConfigs:
    epochs : int = 1
    batch_size : int = 32
    descent: Descent = field(default_factory=Adam)
    save_folderpath = os.path.expanduser(f'~/.py_thunder')
    save_on_done : bool = True
    save_on_epoch : bool = True
    project_name : str = 'unnamed_project'
    enable_logging : bool = False


    def make_wandb_logger(self) -> WBLogger:
        config = {
            'lr': self.descent.lr,
            'batch_size': self.batch_size,
            'optimizer': self.descent.get_algorithm().__name__,
            'epochs': self.epochs,
            'model_architecture': 'unnamed architecture',
            'dataset': 'unnamed dataset',
            'experiment_name': 'unnamed experiment',
            'step_metric' : 'epoch'
        }
        log_dirpath = os.path.expanduser(path='~/.wb_logs')
        wandb_run = wandb.init(project=self.project_name, config=config, dir=log_dirpath)
        return WBLogger(run=wandb_run)


class WBLogger:
    def __init__(self, run : Run):
        self.run : Run = run
        self.current_step : int = 0

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def increment_step(self):
        self.current_step += 1

    @classmethod
    def wandb_is_available(cls) -> bool:
        if os.getenv('WANDB_API_KEY'):
            return True
        elif os.path.isfile(os.path.expanduser('~/.netrc')):
            return True
        else:
            return False