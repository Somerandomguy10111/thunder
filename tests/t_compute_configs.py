import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from torchvision import datasets, transforms

from thunder.module import RunConfigs, ComputeConfigs

# ---------------------------------------------------------

class TestComputeParams(Unittest):
    @classmethod
    def setUpClass(cls):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cls.mnist_train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        cls.mnist_test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)

    def test_thunder_compute_params(self):
        compute_configs = ComputeConfigs(num_gpus=1, dtype=torch.float64)
        mlp = MnistMLP(compute_configs=compute_configs)
        self.assertEqual(mlp.dtype, torch.float64)
        self.assertEqual(mlp.device.type, compute_configs.device.type)


    def test_training(self):
        mlp = MnistMLP()
        run_configs = RunConfigs(epochs=1)
        mlp.train_on(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)

    #
    # def test_non_default_training(self):
    #     compute_configs = ComputeConfigs(num_gpus=1, dtype=torch.float64)
    #     mlp = MnistMLP(compute_configs=compute_configs)
    #     print(f'MLP device is {mlp.device}')
    #     print(f'MLP dtype is {mlp.dtype}')
    #
    #     run_configs = RunConfigs(epochs=1)
    #     mlp.train_on(train_data=self.mnist_train,run_configs=run_configs)
    #

if __name__ == '__main__':
    TestComputeParams.execute_all()