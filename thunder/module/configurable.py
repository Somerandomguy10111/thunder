from __future__ import annotations

from abc import abstractmethod

import torch
from GPUtil import GPU
from torch import device as torchdevice
from torch import dtype as torchdtype
from torch.utils.data import DataLoader, Dataset

from thunder.configs.compute import ComputeConfigs
from thunder.logging import thunderLogger

import GPUtil

# ---------------------------------------------------------

class ComputeManaged(torch.nn.Module):
    def __init__(self, compute_configs : ComputeConfigs = ComputeConfigs()):
        super().__init__()
        self.set_compute_defaults(compute_configs)
        self.compute_configs : ComputeConfigs = compute_configs
        self.gpus : list[GPU] = GPUtil.getGPUs()[:self.compute_configs.num_gpus]
        self.__set__model__()
        self.to(dtype=compute_configs.dtype, device=compute_configs.device)
        print(f'Model device, dtype = {self.compute_configs.device}, {self.compute_configs.dtype}')

    def set_compute_defaults(self, compute_configs : ComputeConfigs):
        target_device, target_dtype = compute_configs.device, compute_configs.dtype

        thunderLogger.warning(f'[Thunder module {self.get_name()}]: Global default torch device set to {target_device}')
        torch.set_default_device(device=target_device)
        thunderLogger.warning(f'[Thunder module {self.get_name()}]: Global default torch dtype set to {target_dtype}')
        torch.set_default_dtype(d=target_dtype)

    @abstractmethod
    def __set__model__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    # ---------------------------------------------------------
    # training routine

    def to_thunder_dataset(self, dataset : Dataset) -> ThunderDataset:
        return ThunderDataset(dataset=dataset, device=self.device, dtype=self.dtype)

    def make_dataloader(self, dataset : Dataset, batch_size : int) -> DataLoader:
        rng = torch.Generator(device=str(self.device))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=rng)

    # ---------------------------------------------------------
    # properties

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

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
