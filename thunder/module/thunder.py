import os.path
from typing import Optional, Union, Dict, Any
from abc import abstractmethod
from datetime import datetime
import pickle
import traceback
import linecache
from holytools.logging import make_logger

import torch
import pytorch_lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import Logger
from torch import Tensor, float32
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from .configs import WBConfig, ComputeConfigs, RunConfigs, ThunderConfig
from .descent import Descent, Adam
from .viewer import Viewer
# ---------------------------------------------------------

thunderLogger = make_logger()

class Thunder(LightningModule):
    def __init__(self, descent : Descent = Adam(),
                       compute_configs : ComputeConfigs = ComputeConfigs(),
                       monitor : Optional[Viewer] = None):
        super().__init__()
        self.descent : Descent = descent
        self.compute_configs : ComputeConfigs = compute_configs
        self.viewer : Optional[Viewer] = monitor
        self.update_state()
        self.__set__model__()

    def update_state(self):
        if not self.compute_configs.dtype == float32:
            print(f'[Thunder module {self.get_name()}: Global default torch dtype set to {self.compute_configs.dtype}')
            self.to(self.compute_configs.dtype)

    @abstractmethod
    def __set__model__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    # ---------------------------------------------------------

    def train_on(self, train_data: Union[Dataset, DataLoader],
                 val_data: Optional[Union[Dataset, DataLoader]] = None,
                 run_configs : RunConfigs = RunConfigs()):
        if self.viewer and val_data:
            self.viewer.sample_batch = self._sample_viewing_batch(test_dataloader=val_data)

        kwargs = {'accelerator' : self.compute_configs.get_accelerator(),
                  'logger' : self.get_logger(run_configs=run_configs),
                  'devices' : self.compute_configs.num_gpus,
                  'max_epochs' : run_configs.epochs,
                  'callbacks' : self.get_callbacks(run_configs=run_configs)}
        pl_trainer = Trainer(**kwargs)
        train_data = self.get_dataloader(data=train_data, run_configs=run_configs)
        val_data = self.get_dataloader(data=val_data, run_configs=run_configs)

        err = None
        try:
            pl_trainer.fit(model=self, train_dataloaders=train_data, val_dataloaders=val_data)
        except Exception as e:
            self.log_relevant_stacktrace(e)
            err = e
        if err:
            if run_configs.print_full_stacktrace:
                raise err
            else:
                raise Exception(f'Encountered excpetion during training routine. Aborting ...')

    def get_callbacks(self, run_configs : RunConfigs) -> list[Callback]:
        callbacks = []
        if run_configs.checkpoint_on_epoch:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kwargs = {'dirpath' : run_configs.save_folderpath, 'filename' : f'{self.get_name()}_{timestamp}'}
            checkpoint_callback = ModelCheckpoint(**kwargs,save_top_k=1, every_n_epochs=1)
            callbacks.append(checkpoint_callback)
        return callbacks


    @staticmethod
    def get_dataloader(data : Union[Dataset, DataLoader], run_configs : RunConfigs):
        if isinstance(data, Dataset):
            data = DataLoader(data, batch_size=run_configs.batch_size)
        return data


    def training_step(self, batch : Tensor, batch_idx):
        x, y = batch
        dtypes_match = x.dtype == y.dtype == self.compute_configs.dtype
        if not dtypes_match:
            raise ValueError(f'Batch x,y dtypes = \"{x.dtype}\",\"{y.dtype}\" but model dtype is \"{self.dtype}\"')

        loss = self.get_loss(predicted=self(x), target=y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self) -> Optimizer:
        params = self.parameters()
        return self.descent.get_optimizer(params=params)


    def on_train_epoch_end(self) -> None:
        if not self.viewer:
            return

        batch = self.viewer.sample_batch
        x,y = batch
        x,y = x.to(self.device), y.to(self.device)
        viewer = self.viewer
        viewer.view(batch=batch, output=self(x))

    @abstractmethod
    def get_loss(self, predicted : Tensor, target : Tensor) -> Tensor:
        pass

    # ---------------------------------------------------------
    # save/load

    @classmethod
    def load(cls, checkpoint_path: str):
        thunder = cls.load_from_checkpoint(checkpoint_path)
        return thunder

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for key, value in checkpoint['thunder_configs'].items():
            value = pickle.loads(value)
            self.__setattr__(name=key, value=value)
        self.update_state()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        thunder_configs = {}
        for key, value in self.__dict__.items():
            if isinstance(value,ThunderConfig):
                thunder_configs[key] = pickle.dumps(value)
        checkpoint['thunder_configs'] = thunder_configs

    # ---------------------------------------------------------
    # logging/viewing

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    def get_logger(self, run_configs: RunConfigs) -> Logger:
        kwargs = {
            "lr": self.descent.lr,
            "batch_size": run_configs.batch_size,
            "epochs": run_configs.epochs,
            "optimizer": self.descent.get_algorithm().__name__,
            "seed": run_configs.seed
        }
        logger = WBConfig(**kwargs).get_logger()
        if not os.path.isdir(logger.save_dir):
            os.makedirs(logger.save_dir)
        return logger


    @staticmethod
    def _sample_viewing_batch(test_dataloader : DataLoader) -> Tensor:
        for batch in test_dataloader:
            return batch

    @staticmethod
    def log_relevant_stacktrace(err : Exception):
        err_class, err_instance, err_traceback = err.__class__, err, err.__traceback__
        tb_list = traceback.extract_tb(err_traceback)

        def is_relevant(tb):
            not_lightning = not os.path.dirname(pytorch_lightning.__file__) in tb.filename
            not_torch = not os.path.dirname(torch.__file__) in tb.filename
            return not_lightning and not_torch

        relevant_tb = [tb for tb in tb_list if is_relevant(tb)]

        if relevant_tb:
            err_msg = "\nEncountered error during training routine. Relevant stacktrace:"
            for frame in relevant_tb:
                file_path = frame.filename
                line_number = frame.lineno
                tb_str = (f'File "{file_path}", line {line_number}, in {frame.name}\n'
                          f'    {linecache.getline(file_path, line_number).strip()}')
                err_msg += f'\n{err_class.__name__}: {err_instance}\n{tb_str}'
            thunderLogger.critical(msg=err_msg)