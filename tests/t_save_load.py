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

        empty_dataloader = DataLoader(EmptyDataset(),batch_size=1)

        compute_configs = ComputeConfigs(dtype=torch.float64)
        descent = SGD(momentum=0.01)
        original_model = MnistMLP(compute_configs=compute_configs, descent=descent)


        short_uuid = str(uuid.uuid4())[:5]
        save_path = f'/tmp/py_thunder_/test_mlp_{short_uuid}.ckpt'
        save_checkpoint = ModelCheckpoint(dirpath=save_path, every_n_epochs=1)

        trainer = Trainer(callbacks=[save_checkpoint], logger=False)
        trainer.fit(model=original_model, train_dataloaders=empty_dataloader)
        trainer.save_checkpoint(filepath=save_path)

        new_model = MnistMLP.load(checkpoint_path=save_path)

        keys_to_remove = ['_trainer', '_hparams']
        a = {k: v for k, v in original_model.__dict__.items() if k not in keys_to_remove}
        b = {k: v for k, v in new_model.__dict__.items() if k not in keys_to_remove}

        print(f'Original state dict: {a}')
        print(f'New state dict     : {b}')
        self.assertEqual(str(a),str(b))

if __name__ == "__main__":
    TestThunderModel.execute_all()