import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from torchvision import datasets, transforms

from thunder.module import RunConfigs, ComputeConfigs

# ---------------------------------------------------------

class TestWBLogging(Unittest):
    @classmethod
    def setUpClass(cls):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cls.mnist_train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        cls.mnist_test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)

    def test_wandb_logging(self):
        compute_configs = ComputeConfigs(num_gpus=0, dtype=torch.float64)
        run_configs = RunConfigs(epochs=1, enable_logging=True)
        mlp = MnistMLP(compute_configs=compute_configs)
        mlp.do_training(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)



if __name__ == '__main__':
    TestWBLogging.execute_all()
