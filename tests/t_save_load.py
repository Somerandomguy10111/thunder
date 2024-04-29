import uuid

import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from thunder.module import ComputeConfigs, SGD, Adam, RunConfigs

class TestThunderModel(Unittest):

    def test_thunder_model_round_trip(self):
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0  # No data

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")


        fpath = f'/tmp/py_thunder_/test_{uuid.uuid4()}.ckpt'
        empty_dataloader = EmptyDataset()
        original = MnistMLP()
        original.train_on(train_data=empty_dataloader)
        original.save(fpath)

        new = MnistMLP.load(fpath=fpath)

        # new_model = MnistMLP.load(checkpoint_path=save_path)
        #
        # keys_to_remove = ['_trainer', '_hparams']
        # a = {k: v for k, v in original_model.__dict__.items() if k not in keys_to_remove}
        # b = {k: v for k, v in new_model.__dict__.items() if k not in keys_to_remove}
        #
        # print(f'Original state dict: {a}')
        # print(f'New state dict     : {b}')
        # self.assertEqual(str(a),str(b))

if __name__ == "__main__":
    TestThunderModel.execute_all()