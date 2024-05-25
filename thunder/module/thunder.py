import os.path
from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .configs import ComputeConfigs, RunConfigs, ComputeConformDataset, WBLogger
from holytools.logging import LoggerFactory

thunderLogger = LoggerFactory.make_logger(name=__name__)
# ---------------------------------------------------------


class Thunder(torch.nn.Module):
# class Thunder:
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

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    # ---------------------------------------------------------
    # training routine

    def do_training(self, train_data: Dataset, val_data: Optional[Dataset] = None,
                    run_configs : RunConfigs = RunConfigs()):
        batch_size = run_configs.batch_size
        train_loader = self.get_dataloader(dataset=train_data, batch_size=batch_size)
        # val_loader = self.get_dataloader(dataset=val_data, batch_size=batch_size) if val_data else None
        optimizer = run_configs.descent.get_optimizer(params=self.parameters())
        max_epochs = run_configs.epochs
        model = nn.DataParallel(self) if self.compute_configs.num_gpus > 1 else self
        if run_configs.enable_logging:
            self.wblogger = run_configs.make_wandb_logger()

        err = None
        try:
            self.train()
            for epoch in range(max_epochs):
                self.train_epoch(train_loader=train_loader, optimizer=optimizer, model=model)
                self.validate_epoch()
                if run_configs.save_on_epoch:
                    self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_{epoch}.pth')
        except Exception as e:
            err = e
        if err:
            print(f'Encountered exception during training routine. Aborting ...')
            raise err

        if run_configs.save_on_done:
            self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_final.pth')


    def get_dataloader(self, dataset : Dataset, batch_size : int) -> DataLoader:
        compute_conform_dataset = ComputeConformDataset(dataset, self.compute_configs.device, self.compute_configs.dtype)
        return DataLoader(compute_conform_dataset, batch_size=batch_size)


    def train_epoch(self, train_loader : DataLoader, optimizer : torch.optim.Optimizer, model : nn.Module):
        for j, batch in enumerate(train_loader):
            inputs, labels = batch
            loss = self.get_loss(predicted=model(inputs), target=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            self.wblogger.increment_step()
            self.log_loss(loss=loss)

    def validate_epoch(self):
        pass

    def log_loss(self, loss):
        if not self.wblogger is None:
            self.wblogger.log({'loss': loss}, step=self.wblogger.current_step)


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
    # view

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