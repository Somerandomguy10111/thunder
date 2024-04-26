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


        save_path = '/tmp/py_thunder_/test_mlp.ckpt'
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


    def test_training(self):
        class CustomMNIST(datasets.MNIST):
            def __getitem__(self, index):
                x, y = super(CustomMNIST, self).__getitem__(index)
                return x, y

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train = CustomMNIST('/tmp/mnist', train=True, download=True, transform=transform)
        mnist_test = CustomMNIST('/tmp/mnist', train=False, download=True, transform=transform)


        test_x, test_y = mnist_test[0]
        self.assertEqual(test_x.dtype, torch.float32)
        self.assertIsInstance(test_y, int)
        train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=3)
        val_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

        # monitor.set_viewer(viewer=MNISTViewer())
        mlp = MnistMLP(descent=Adam(lr=0.001))
        run_configs = RunConfigs(epochs=1)
        mlp.train_on(train_data=train_loader, val_data=val_loader, run_configs=run_configs)


if __name__ == '__main__':
    TestThunderModel.execute_all()
