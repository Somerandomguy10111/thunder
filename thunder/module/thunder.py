import copy
import os.path
from abc import abstractmethod
from typing import Optional, Callable

import torch
from holytools.userIO import TrackedInt
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from thunder.configs import RunConfigs, ComputeConfigs, Devices
from thunder.logging import Metric, WBLogger, thunderLogger
from .managed import ComputeManaged


# ---------------------------------------------------------

class Thunder(ComputeManaged):
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
            self.wblogger = run_configs.get_wandb_logger()

        train_model = nn.DataParallel(self) if self.compute_configs.num_gpus > 1 else self
        optimizer = run_configs.descent.get_optimizer(params=self.parameters())
        thunderLogger.info(msg=f'[Thunder module {self.get_name()}]: Starting training')
        for epoch in range(run_configs.epochs):
            thunderLogger.info(f'[Thunder module {self.get_name()}]: Training epoch number {epoch}...')
            self.train_epoch(train_loader=train_loader, optimizer=optimizer, model=train_model)
            if val_loader:
                self.validate_epoch(val_loader=val_loader)
            if run_configs.save_on_epoch:
                self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_{epoch}.pth')
        if run_configs.save_on_done:
            self.save(fpath=f'{run_configs.save_folderpath}/{self.get_name()}_final.pth')
        if not self.wblogger is None:
            self.wblogger.finish_subrun()


    # ---------------------------------------------------------
    # optimization

    def train_epoch(self, train_loader : DataLoader, optimizer : torch.optim.Optimizer, model : nn.Module):
        self.train()

        min_batches = max(len(train_loader),1)
        tracked_int = TrackedInt(start_value=0, finish_value=min_batches)

        for batch in train_loader:
            inputs, labels = batch
            loss = self.get_loss(predicted=model(inputs), target=labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tracked_int.increment(to_add=1)
            if not self.wblogger is None:
                self.wblogger.increment_batch()
                self.wblogger.log_quantity(name='batch', value=self.wblogger.current_batch)
                self.log_compute_resources()

        if not tracked_int.progressbar.finished():
            tracked_int.finish()
        if not self.wblogger is None:
            self.wblogger.increment_epoch()
            self.wblogger.log_quantity(name='epoch', value=self.wblogger.current_epoch)
            self.log_batch_metrics(is_training=True)


    def validate_epoch(self, val_loader : DataLoader):
        self.eval()
        val_loss = 0
        for batch in val_loader:
            inputs, labels = batch
            loss = self.get_loss(predicted=self(inputs), target=labels)
            val_loss += loss.item()
        if not self.wblogger is None:
            self.log_batch_metrics(is_training=False)

    @abstractmethod
    def get_loss(self, predicted : Tensor, target : Tensor) -> Tensor:
        pass

    # ---------------------------------------------------------
    # logging

    def log_compute_resources(self):
        if self.compute_configs.device == Devices.gpu:
            self.wblogger.log_gpu_resources(gpus=self.gpus)

    def log_batch_metrics(self, is_training : bool):
        self.wblogger.log_metrics(metric_map=self.metric_map, is_training=is_training)
        self.metric_map = {}

    @staticmethod
    def add_metric(name_override : Optional[str] = None, report_average : bool = False, add_modelname : bool = False):

        def add_metric_decorator(mthd : Callable[..., Tensor | float | list[float]]):
            def logged_mthd(self: Thunder, *args, **kwargs):
                result = mthd(self, *args, **kwargs)
                metric_name = name_override if not name_override is None else mthd.__name__
                if add_modelname:
                    metric_name = f'{self.get_name()}_{metric_name}'

                try:
                    logged_values = copy.copy(result)
                    if isinstance(logged_values, Tensor):
                        if logged_values.dim() > 1:
                            raise ValueError(f'Can only log 0 axis (scalars) or 1 axis tensors (vectors).'
                                             f' Metric \"{metric_name}\" has {logged_values.dim()} axes')
                        logged_values = logged_values.tolist()
                    if isinstance(logged_values, list):
                        logged_values = [float(x) for x in logged_values]
                    if isinstance(logged_values, float):
                        logged_values = [logged_values]
                    logged_values: list[float]

                    if not metric_name in self.metric_map:
                        self.metric_map[metric_name] = Metric(log_average=report_average)
                    self.metric_map[metric_name].add(new_values=logged_values)

                except Exception as e:
                    thunderLogger.warning(f'Failed to log metric \"{metric_name}\": {e}')

                return result
            return logged_mthd

        return add_metric_decorator
