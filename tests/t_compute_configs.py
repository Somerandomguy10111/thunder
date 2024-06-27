import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from torchvision import datasets, transforms

from thunder.module import RunConfigs, ComputeConfigs

# ---------------------------------------------------------

class TestComputeConfigs(Unittest):
    @classmethod
    def setUpClass(cls):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cls.mnist_train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        cls.mnist_test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
        cls.run_configs = RunConfigs(epochs=1)

        cls.gpu_configs = ComputeConfigs(num_gpus=1, dtype=torch.float64)
        cls.cpu_configs = ComputeConfigs(num_gpus=0, dtype=torch.float64)

    # ---------------------------------------------------------
    # configs

    def test_gpu_configs(self):
        if not torch.cuda.is_available():
            self.skipTest(reason=f'GPU unavailable')

        mlp = MnistMLP(compute_configs=self.gpu_configs)
        print(f'MLP device, dtype = {mlp.device}, {mlp.dtype}')
        self.assertEqual(mlp.dtype, torch.float64)
        self.assertEqual(mlp.device.type, self.gpu_configs.device.type)

    def test_cpu_configs(self):
        mlp = MnistMLP(compute_configs=self.cpu_configs)
        print(f'MLP device, dtype = {mlp.device}, {mlp.dtype}')
        self.assertEqual(mlp.dtype, torch.float64)
        self.assertEqual(mlp.device.type, self.cpu_configs.device.type)

    # ---------------------------------------------------------
    # training

    def test_gpu_training(self):
        if not torch.cuda.is_available():
            self.skipTest(reason=f'GPU unavailable')

        mlp = MnistMLP(compute_configs=self.gpu_configs)
        mlp.do_training(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=self.run_configs)


    def test_cpu_training(self):
        compute_configs = ComputeConfigs(num_gpus=0, dtype=torch.float64)
        mlp = MnistMLP(compute_configs=compute_configs)
        mlp.do_training(train_data=self.mnist_train, run_configs=self.run_configs)


if __name__ == '__main__':
    TestComputeConfigs.execute_all()
