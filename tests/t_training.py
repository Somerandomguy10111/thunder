import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from torchvision import datasets, transforms

from thunder.module import RunConfigs, ComputeConfigs

# ---------------------------------------------------------

class TestThunderTraining(Unittest):
    @classmethod
    def setUpClass(cls):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cls.mnist_train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        cls.mnist_test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)

    # def test_dataset_training(self):
    #     test_x, test_y = self.mnist_test[0]
    #     self.assertEqual(test_x.dtype, torch.float32)
    #     self.assertIsInstance(test_y, int)
    #
    #     mlp = MnistMLP()
    #     run_configs = RunConfigs(epochs=1)
    #     mlp.train_on(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)


    def test_device_mismatch(self):
        compute_configs = ComputeConfigs(num_gpus=1, dtype=torch.float64)
        mlp = MnistMLP(compute_configs=compute_configs)
        print(f'MLP device is {mlp.device}')
        print(f'MLP dtype is {mlp.dtype}')

        run_configs = RunConfigs(epochs=1)
        mlp.train_on(train_data=self.mnist_train,run_configs=run_configs)


    # def test_datatype_mismatch(self):
    #     # Configure model to expect a different datatype
    #     compute_configs = ComputeConfigs(dtype=torch.float64)
    #     mlp = MnistMLP(compute_configs=compute_configs, descent=SGD(momentum=0.01))
    #
    #     mnist_train = self.get_mnist(train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    #     mnist_test = self.get_mnist(train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    #
    #     with self.assertRaises(DatatypeError):
    #         mlp.train_on(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=RunConfigs(epochs=1))
    #
    #
    # def get_mnist(self, train : bool = False, transform):
    #     return datasets.MNIST('/tmp/mnist', train=train, download=True, transform=transform)

if __name__ == '__main__':
    TestThunderTraining.execute_all()
