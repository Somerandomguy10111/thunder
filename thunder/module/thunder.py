import os.path
from abc import abstractmethod
from typing import Optional

from holytools.logging import LoggerFactory
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .configs import ComputeConfigs, RunConfigs, ThunderDataset, WBLogger

thunderLogger = LoggerFactory.make_logger(name=__name__)
# ---------------------------------------------------------


class Thunder(torch.nn.Module):
    def __init__(self, compute_configs : ComputeConfigs = ComputeConfigs()):
        super().__init__()
        self.set_compute_defaults(compute_configs)
        self.wblogger : Optional[WBLogger] = None
        self.compute_configs : ComputeConfigs = compute_configs
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

    def do_training(self, train_data: Dataset,
                          val_data: Optional[Dataset] = None,
                          run_configs : RunConfigs = RunConfigs()):
        to_thunder_dataset = lambda dataset : ThunderDataset(dataset=dataset,
                                                             torch_device=self.compute_configs.device,
                                                             torch_dtype=self.compute_configs.dtype)
        train_data = to_thunder_dataset(dataset=train_data)
        train_loader = self.make_dataloader(dataset=train_data, batch_size=run_configs.batch_size)

        if val_data:
            val_data = to_thunder_dataset(dataset=val_data)
            val_loader = self.make_dataloader(dataset=val_data, batch_size=run_configs.batch_size)
        else:
            val_loader = None
        if run_configs.enable_logging:
            self.wblogger = run_configs.make_wandb_logger()

        self.train()
        train_model = nn.DataParallel(self) if self.compute_configs.num_gpus > 1 else self
        optimizer = run_configs.descent.get_optimizer(params=self.parameters())
        for epoch in range(run_configs.epochs):
            self.train_epoch(train_loader=train_loader, optimizer=optimizer, model=train_model)
            if val_loader:
                self.validate_epoch(val_loader=val_loader)
            if run_configs.save_on_epoch:
                self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_{epoch}.pth')
        if run_configs.save_on_done:
            self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_final.pth')
        self.on_epoch_done()


    def make_dataloader(self, dataset : Dataset, batch_size : int) -> DataLoader:
        rng = torch.Generator(device=str(self.device))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=rng)

    # ---------------------------------------------------------
    # optimization

    def train_epoch(self, train_loader : DataLoader, optimizer : torch.optim.Optimizer, model : nn.Module):
        for j, batch in enumerate(train_loader):
            inputs, labels = batch
            loss = self.get_loss(predicted=model(inputs), target=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            self.wblogger.increment_batch()

        if not self.wblogger is None:
            self.wblogger.increment_epoch()


    def validate_epoch(self, val_loader : DataLoader):
        val_loss = 0
        for batch in val_loader:
            inputs, labels = batch
            loss = self.get_loss(predicted=self(inputs), target=labels)
            val_loss += loss.item()


    @abstractmethod
    def get_loss(self, predicted : Tensor, target : Tensor) -> Tensor:
        pass

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

class DatatypeError(Exception):
    pass