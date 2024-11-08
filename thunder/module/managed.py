from __future__ import annotations

import os
from abc import abstractmethod
from logging import Logger
from typing import Optional

import GPUtil
import torch
from GPUtil import GPU
from torch import device as torchdevice, Tensor
from torch import dtype as torchdtype
from torch.utils.data import DataLoader, Dataset

from holytools.logging import LoggerFactory
from thunder.configs.compute import ComputeConfig


# ---------------------------------------------------------

class ComputeManaged(torch.nn.Module):
    def __init__(self, compute_configs : ComputeConfig = ComputeConfig()):
        super().__init__()
        self.pylogger : Logger = LoggerFactory.get_logger(name=self.get_name(), include_logger_name=True)
        self.compute_configs : ComputeConfig = compute_configs
        self.gpus : list[GPU] = GPUtil.getGPUs()[:self.compute_configs.get_num_gpus()]
        self.set_compute_defaults(compute_configs)
        self.__set__model__()
        self.to(dtype=compute_configs.dtype, device=compute_configs.device)
        self.pylogger.info(f'Model device, dtype = {self.compute_configs.device}, {self.compute_configs.dtype}')

    def set_compute_defaults(self, compute_configs : ComputeConfig):
        target_device, target_dtype = compute_configs.device, compute_configs.dtype

        self.pylogger.warning(f'Global default torch device set to {target_device}')
        torch.set_default_device(device=target_device)
        self.pylogger.warning(f'Global default torch dtype set to {target_dtype}')
        torch.set_default_dtype(target_dtype)
        if compute_configs.allow_tensor_cores:
            self.pylogger.warning(f'Enabling Tensor Cores and TF32')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @abstractmethod
    def __set__model__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    # ---------------------------------------------------------
    # training

    def to_thunder_dataset(self, dataset : Dataset) -> ThunderDataset:
        return ThunderDataset(dataset=dataset, device=self.device, dtype=self.dtype)

    def make_dataloader(self, dataset : Dataset, batch_size : int) -> DataLoader:
        rng = torch.Generator(device=str(self.device))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=rng)

    # ---------------------------------------------------------
    # save/load

    @classmethod
    def load(cls, fpath: str):
        checkpoint = torch.load(fpath)
        model = cls(compute_configs=checkpoint['compute_configs'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


    def save(self, fpath : str):
        save_fpath = os.path.abspath(os.path.relpath(fpath))
        save_dirpath = os.path.dirname(save_fpath)
        os.makedirs(save_dirpath, exist_ok=True)

        checkpoint = {
            'state_dict': self.state_dict(),
            'compute_configs': self.compute_configs
        }
        torch.save(checkpoint, fpath)

    # ---------------------------------------------------------
    # properties

    @classmethod
    def get_name(cls) -> str:
        return f'Thunder module {cls.__name__}'

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self):
        try:
            param = next(self.parameters())
        except StopIteration:
            param = next(self.buffers())
        return param.device

    @property
    def dtype(self):
        try:
            param = next(self.parameters())
        except StopIteration:
            param = next(self.buffers())
        return param.dtype


class ThunderDataset(Dataset):
    def __init__(self, dataset : Dataset, device : torchdevice, dtype : torchdtype):
        self.base_dataset : Dataset = dataset
        self.device : device = device
        self.dtype : dtype = dtype

    def __len__(self) -> Optional[int]:
        if hasattr(self.base_dataset, '__len__'):
            # noinspection PyTypeChecker
            return len(self.base_dataset)
        else:
            return None

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            x, y = item
            if isinstance(x, torch.Tensor):
                x = self.to_conform_tensor(x)
            if isinstance(y, torch.Tensor):
                y = self.to_conform_tensor(y)
            return x, y
        else:
            item = item

        return item

    # ---------------------------------------------------------


    def to_conform_tensor(self, t : Tensor):
        if self.is_compute_conform(tensor=t):
            return t
        # print(f'Compute not conform, converting ...')
        return t.to(dtype=self.dtype, device=self.device)


    def is_compute_conform(self, tensor : Tensor) -> bool:
        # print(f'self dtype, deviice = {self.dtype}, {self.device}')
        # print(f'tensor dtype, device = {tensor.dtype}, {tensor.device}')
        if tensor.dtype != self.dtype:
            return  False
        if tensor.device != self.device:
            return False
        return True