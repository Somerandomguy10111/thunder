from __future__ import annotations
import os
import torch

from dataclasses import asdict
from pytorch_lightning.loggers import Logger, WandbLogger

from dataclasses import dataclass
from torch import device, dtype
from typing import Optional
# ---------------------------------------------------------

class ThunderConfig:
    pass

@dataclass
class ComputeConfigs(ThunderConfig):
    torch_device : device = device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    dtype : dtype = torch.float32

    def get_accelerator(self) -> str:
        return "gpu" if self.num_gpus > 0 else "cpu"


@dataclass
class RunConfigs:
    epochs : int = 1
    batch_size : int = 32
    seed : int = 42
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
