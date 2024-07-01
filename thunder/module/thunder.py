import copy
import os.path
from abc import abstractmethod
from typing import Optional, Callable

import torch
from tabulate import tabulate
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from thunder.configs import RunConfigs, ComputeConfigs
from thunder.logging import Metric, WBLogger
from .configurable import ComputeConfigurable

# ---------------------------------------------------------

class Thunder(ComputeConfigurable):
    def __init__(self, compute_configs : ComputeConfigs = ComputeConfigs()):
        super().__init__(compute_configs=compute_configs)
        self.wblogger : Optional[WBLogger] = None
        self.metric_map : dict[str, Metric] = {}
        self.__set__model__()
        self.to(dtype=compute_configs.dtype, device=compute_configs.device)


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
        train_data = self.to_thunder_dataset(dataset=train_data)
        train_loader = self.make_dataloader(dataset=train_data, batch_size=run_configs.batch_size)

        if val_data:
            val_data = self.to_thunder_dataset(dataset=val_data)
            val_loader = self.make_dataloader(dataset=val_data, batch_size=run_configs.batch_size)
        else:
            val_loader = None
        if run_configs.enable_logging:
            self.wblogger = run_configs.make_wandb_logger()

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


    # ---------------------------------------------------------
    # optimization

    def train_epoch(self, train_loader : DataLoader, optimizer : torch.optim.Optimizer, model : nn.Module):
        self.train()
        for j, batch in enumerate(train_loader):
            inputs, labels = batch
            loss = self.get_loss(predicted=model(inputs), target=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if not self.wblogger is None:
                self.wblogger.increment_batch()
                self.wblogger.log_quantity(name='batch', value=self.wblogger.current_batch)

        if not self.wblogger is None:
            self.wblogger.increment_epoch()
            self.wblogger.log_quantity(name='epoch', value=self.wblogger.current_epoch)
            self.log_metrics(is_training=True)


    def validate_epoch(self, val_loader : DataLoader):
        self.eval()
        val_loss = 0
        for batch in val_loader:
            inputs, labels = batch
            loss = self.get_loss(predicted=self(inputs), target=labels)
            val_loss += loss.item()
        if not self.wblogger is None:
            self.log_metrics(is_training=False)

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
    # logging

    def log_metrics(self, is_training : bool):
        table_data = []
        for k,v in self.metric_map.items():
            if is_training:
                self.wblogger.log_training_quantity(name=k, value=v.value)
            else:
                self.wblogger.log_validation_quantity(name=k, value=v.value)
            table_data.append([k, v.value])
        self.metric_map = {}
        table = tabulate(table_data, headers=['Metric', 'Value'], tablefmt='psql')
        print(f'Epoch {self.wblogger.current_epoch} {"Training" if is_training else "Validation"} metrics:')
        print(table)
        print()

    @staticmethod
    def add_metric(mthd : Callable[..., Tensor | float | list[float]], name_override : Optional[str] = None, log_average : bool = False):
        metric_name = name_override if not name_override is None else mthd.__name__

        def logged_mthd(self : Thunder, *args, **kwargs):
            result = mthd(self, *args, **kwargs)

            logged_values = copy.copy(result)
            if isinstance(logged_values, Tensor):
                logged_values = logged_values.tolist()
                logged_values = [float(x) for x in logged_values]
            if isinstance(logged_values, list):
                if not all([isinstance(v, float) for v in logged_values]):
                    raise ValueError(f'Metric {mthd.__name__} did not return a list of floats')
            if isinstance(logged_values, float):
                logged_values = [logged_values]
            logged_values : list[float]

            if not mthd.__name__ in self.metric_map:
                self.metric_map[metric_name] = Metric(log_average=log_average)
            self.metric_map[metric_name].add(new_values=logged_values)

            return result

        return logged_mthd
