import uuid

import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from thunder.module import ComputeConfigs, SGD, Adam, RunConfigs

# ---------------------------------------------------------

class TestThunderModel(Unittest):
    @classmethod
    def setUpClass(cls):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cls.mnist_train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        cls.mnist_test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)

    def test_dataset_training(self):
        test_x, test_y = self.mnist_test[0]
        self.assertEqual(test_x.dtype, torch.float32)
        self.assertIsInstance(test_y, int)

        mlp = MnistMLP(descent=Adam(lr=0.001))
        run_configs = RunConfigs(epochs=1)
        mlp.train_on(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)


    def test_dataloader_training(self):
        mlp = MnistMLP(descent=Adam(lr=0.001))
        run_configs = RunConfigs(epochs=1)
        train_loader = DataLoader(self.mnist_train, batch_size=64, shuffle=True, num_workers=3)
        val_loader = DataLoader(self.mnist_test, batch_size=64, shuffle=False)
        mlp.train_on(train_data=train_loader, val_data=val_loader, run_configs=run_configs)


    def test_device_mismatch(self):
        class DeviceTransformDataLoader(DataLoader):
            def __init__(self, dataset, batch_size, shuffle, the_device):
                super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
                self.device = the_device

            def collate_fn(self, batch):
                tensors, targets = zip(*batch)
                tensors = torch.stack(tensors).to(self.device)
                targets = torch.tensor(targets).to(self.device)
                return tensors, targets


        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrong_device = "cuda" if device == "cpu" else "cpu"

        compute_configs = ComputeConfigs(torch_device=torch.device(device))
        mlp = MnistMLP(compute_configs=compute_configs, descent=SGD(momentum=0.01))
        train_loader_wrong_device = DeviceTransformDataLoader(self.mnist_train, batch_size=64, shuffle=True, the_device=wrong_device)
        with self.assertRaises(Exception):
            mlp.train_on(train_data=train_loader_wrong_device, run_configs=RunConfigs(epochs=1))


    def test_datatype_mismatch(self):
        # Configure model to expect a different datatype
        compute_configs = ComputeConfigs(dtype=torch.float64)
        mlp = MnistMLP(compute_configs=compute_configs, descent=SGD(momentum=0.01))

        # Setup DataLoader to provide data in the incorrect datatype
        def to_incorrect_dtype(batch):
            x, y = batch
            return x.float(), y.long()  # Intentionally wrong datatype

        train_loader = DataLoader(self.mnist_train, batch_size=64, shuffle=True, collate_fn=to_incorrect_dtype)
        val_loader = DataLoader(self.mnist_test, batch_size=64, shuffle=False, collate_fn=to_incorrect_dtype)

        with self.assertRaises(Exception):
            mlp.train_on(train_data=train_loader, val_data=val_loader, run_configs=RunConfigs(epochs=1))


if __name__ == '__main__':
    TestThunderModel.execute_all()
