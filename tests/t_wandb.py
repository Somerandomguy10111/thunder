import torch
from holytools.devtools import Unittest
from torch import device

from tests.mnist import MnistMLP, MnistMLPVariation
from torchvision import datasets, transforms


from torch.utils.data import random_split

from thunder.configs import RunConfig, ComputeConfig
from thunder.logging import WBLogger


# ---------------------------------------------------------

class TestWBLogging(Unittest):
    def setUp(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        mnist_train_full = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        mnist_test_full = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)

        fraction = 40
        train_target_length = len(mnist_train_full) // fraction
        test_target_length = len(mnist_test_full) // fraction
        generator = torch.Generator(device='cuda')
        self.mnist_train, _ = random_split(mnist_train_full, [train_target_length, len(mnist_train_full) - train_target_length], generator=generator)
        self.mnist_test, _ = random_split(mnist_test_full, [test_target_length, len(mnist_test_full) - test_target_length], generator=generator)


    def test_wandb_logging(self):
        if not WBLogger.wandb_is_available():
            self.skipTest(reason=f'Wandb is unavailable')

        run_configs = RunConfig(epochs=2, enable_wandb=True, run_name=f't_wandb_run', batch_size=16)
        compute_configs = ComputeConfig(device=device(f'cuda:0'), dtype=torch.float64)
        mlp = MnistMLP(compute_configs=compute_configs)
        mlp.do_training(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)

        mlp2 = MnistMLPVariation(compute_configs=compute_configs)
        mlp2.do_training(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)

if __name__ == '__main__':
    TestWBLogging.execute_all()
