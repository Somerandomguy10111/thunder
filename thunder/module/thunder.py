from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .configs import ComputeConfigs, RunConfigs, ComputeConformDataset
from .logging import log_relevant_stacktrace, thunderLogger


# ---------------------------------------------------------

class Thunder(torch.nn.Module):
    def __init__(self, compute_configs : ComputeConfigs = ComputeConfigs()):
        super().__init__()
        self.set_compute_defaults(compute_configs)
        self.compute_configs : ComputeConfigs = compute_configs
        self.__set__model__()
        self.to(dtype=compute_configs.dtype, device=compute_configs.device)
        print(f'Model device, dtype = {self.device}, {self.dtype}')

    def set_compute_defaults(self, compute_configs : ComputeConfigs):
        target_device = compute_configs.device
        target_dtype = compute_configs.dtype
        
        thunderLogger.warn(f'[Thunder module {self.get_name()}]: Global default torch device set to {target_device}')
        torch.set_default_device(device=target_device)
        thunderLogger.warn(f'[Thunder module {self.get_name()}]: Global default torch dtype set to {target_dtype}')
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

    def train_on(self, train_data: Dataset, val_data: Optional[Dataset] = None,
                       run_configs : RunConfigs = RunConfigs()):
        batch_size = run_configs.batch_size
        train_loader = self.get_dataloader(dataset=train_data, batch_size=batch_size)
        val_loader = self.get_dataloader(dataset=val_data, batch_size=batch_size) if val_data else None
        optimizer = run_configs.descent.get_optimizer(params=self.parameters())
        max_epochs = run_configs.epochs

        try:
            self.train()
            for epoch in range(max_epochs):
                self.epoch_training(train_loader=train_loader, optimizer=optimizer)
                self.epoch_validation()
                if run_configs.save_on_epoch:
                    self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_{epoch}.pth')
        except Exception as e:
            log_relevant_stacktrace(e)
            if run_configs.print_full_stacktrace:
                raise e
            else:
                raise Exception('Encountered exception during training routine. Aborting ...')

        if run_configs.save_on_done:
            self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_final.pth')


    def get_dataloader(self, dataset : Dataset, batch_size : int) -> DataLoader:
        compute_conform_dataset = ComputeConformDataset(dataset, self.device, self.dtype)
        return DataLoader(compute_conform_dataset, batch_size=batch_size)


    def epoch_training(self, train_loader : DataLoader, optimizer ):
        for batch in train_loader:
            inputs, labels = batch
            loss = self.get_loss(predicted=self(inputs), target=labels)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


    def epoch_validation(self):
        pass

    @abstractmethod
    def get_loss(self, predicted : Tensor, target : Tensor) -> Tensor:
        pass

    # ---------------------------------------------------------
    # save/load

    @classmethod
    def load(cls, fpath: str):
        checkpoint = torch.load(fpath)
        kwargs = checkpoint['compute_configs']
        model = cls(**kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        return model


    def save(self, fpath : str):
        checkpoint = {
            'state_dict': self.state_dict(),
            'compute_configs': self.compute_configs
        }
        torch.save(checkpoint, fpath)


class DatatypeError(Exception):
    pass
