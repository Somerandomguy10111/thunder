import torch
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from torchvision import datasets, transforms

from thunder.module import RunConfigs, ComputeConfigs
from thunder.module.configs import WBLogger


# ---------------------------------------------------------

class TestWBLogging(Unittest):
    @classmethod
    def setUpClass(cls):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        cls.mnist_train = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        cls.mnist_test = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)

    def test_wandb_logging(self):
        if not WBLogger.wandb_is_available():
            self.skipTest(reason=f'Wandb is unavailable')

        compute_configs = ComputeConfigs(num_gpus=0, dtype=torch.float64)
        run_configs = RunConfigs(epochs=1, enable_logging=True)
        mlp = MnistMLP(compute_configs=compute_configs)
        mlp.do_training(train_data=self.mnist_train, val_data=self.mnist_test, run_configs=run_configs)


if __name__ == '__main__':
    import wandb
    # TestWBLogging.execute_all()
    print(WBLogger.wandb_is_available())
