from typing import Optional, Dict, Any
from abc import abstractmethod
from datetime import datetime
import pickle
import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from .configs import ComputeConfigs, RunConfigs, ThunderConfig
from .viewer import Viewer
from .logging import log_relevant_stacktrace, get_wb_logger, thunderLogger

# ---------------------------------------------------------

class Thunder(LightningModule):
    def __init__(self, compute_configs : ComputeConfigs = ComputeConfigs(), viewer : Optional[Viewer] = None):
        super().__init__()
        self.set_compute_defaults(compute_configs)
        self.compute_configs : ComputeConfigs = compute_configs
        self.viewer : Optional[Viewer] = viewer
        self.optimizer : Optional[Optimizer] = None
        self.trainer : Optional[Trainer] = None
        self.__set__model__()

    def set_compute_defaults(self, compute_configs : ComputeConfigs):
        target_device = compute_configs.get_device()
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

    def train_on(self, train_data: Dataset,
                       val_data: Optional[Dataset] = None,
                       run_configs : RunConfigs = RunConfigs()):
        if self.viewer and not val_data is None:
            self.viewer.sample = train_data[0]

        kwargs = {'accelerator' : self.compute_configs.get_accelerator(),
                  'logger' : get_wb_logger(run_configs=run_configs),
                  'devices' : self.compute_configs.num_gpus,
                  'max_epochs' : run_configs.epochs,
                  'callbacks' : self.get_callbacks(run_configs=run_configs)}
        pl_trainer = Trainer(**kwargs)
        self.optimizer = run_configs.descent.get_optimizer(params=self.parameters())
        train_data = DataLoader(train_data, batch_size=run_configs.batch_size)
        val_data = DataLoader(val_data, batch_size=run_configs.batch_size) if val_data else None

        err = None
        try:
            pl_trainer.fit(model=self, train_dataloaders=train_data, val_dataloaders=val_data)
        except Exception as e:
            log_relevant_stacktrace(e)
            err = e

        if err:
            err = err if run_configs.print_full_stacktrace else Exception('Encountered excpetion during training routine. Aborting ...')
            raise err


    def get_callbacks(self, run_configs : RunConfigs) -> list[Callback]:
        callbacks = []
        if run_configs.checkpoint_on_epoch:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            kwargs = {'dirpath' : run_configs.save_folderpath, 'filename' : f'{self.get_name()}_{timestamp}'}
            checkpoint_callback = ModelCheckpoint(**kwargs,save_top_k=1, every_n_epochs=1)
            callbacks.append(checkpoint_callback)
        return callbacks


    def training_step(self, batch : Tensor, batch_idx):
        x, y = batch
        dtypes_match = x.dtype == self.dtype == self.compute_configs.dtype
        if not dtypes_match:
            raise DatatypeError(f'Batch input dtype = \"{x.dtype}\", model dtype is \"{self.dtype}\"; '
                             f'Compute configs dictate : \"{self.compute_configs.dtype}\"')

        loss = self.get_loss(predicted=self(x), target=y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        if self.optimizer is None:
            raise ValueError('Optimizer not set. Cannot configure optimizer for trainer fit routine')
        return self.optimizer


    def on_train_epoch_end(self) -> None:
        if not self.viewer:
            return

        batch = self.viewer.sample
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
    def load(cls, fpath: str):
        thunder = cls.load_from_checkpoint(fpath)
        return thunder

    def save(self, fpath : str):
        if self.trainer is None:
            raise ValueError('Trainer is None. Pytorch lightning can only be saved after running trainer.fit(...) on model. Aborting ...')
        if self.trainer.model is None:
            raise ValueError('Trainer model is None. Pytorch lightning can only be saved after running trainer.fit(...) on model. Aborting ...')
        self.trainer.save_checkpoint(filepath=fpath)


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        for key, value in checkpoint['thunder_configs'].items():
            value = pickle.loads(value)
            self.__setattr__(name=key, value=value)
        self.set_compute_defaults(self.compute_configs)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        thunder_configs = {}
        for key, value in self.__dict__.items():
            if isinstance(value,ThunderConfig):
                thunder_configs[key] = pickle.dumps(value)
        checkpoint['thunder_configs'] = thunder_configs



class DatatypeError(Exception):
    pass
