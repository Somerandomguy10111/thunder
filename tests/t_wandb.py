import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from torchvision import datasets, transforms

from thunder import RunConfigs, ComputeConfigs

from torch.utils.data import random_split

from thunder.logging.wblogger import WBLogger


# ---------------------------------------------------------

class TestWBLogging(Unittest):
    def setUp(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train_full = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        mnist_test_full = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)

        fraction = 40
        train_target_length = len(mnist_train_full) // fraction
        test_target_length = len(mnist_test_full) // fraction
        self.mnist_train, _ = torch.utils.data.random_split(mnist_train_full, [train_target_length, len(mnist_train_full) - train_target_length])
        self.mnist_test, _ = torch.utils.data.random_split(mnist_test_full, [test_target_length, len(mnist_test_full) - test_target_length])


    def test_wandb_logging(self):
        if not WBLogger.wandb_is_available():
            self.skipTest(reason=f'Wandb is unavailable')

        compute_configs = ComputeConfigs(num_gpus=0, dtype=torch.float64)
        run_configs = RunConfigs(epochs=2, enable_logging=True)
        mlp = MnistMLP(compute_configs=compute_configs)
        mlp.do_training(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)


if __name__ == '__main__':
    TestWBLogging.execute_all()
