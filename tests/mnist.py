from torch.nn import Linear
from torch import Tensor
from torch.nn.functional import relu
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
import random

from thunder.module import Thunder
from thunder import Viewer


# ---------------------------------------------------------

class MnistMLP(Thunder):
    def __set__model__(self):
        self.fc1 = Linear(28 * 28, 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, 10)

    def forward(self, x : Tensor) -> Tensor:
        y = x.view(-1, 28 * 28)
        y = relu(self.fc1(y))
        y = relu(self.fc2(y))
        y = self.fc3(y)
        return y

    def get_loss(self, predicted : Tensor, target : Tensor) -> Tensor:
        loss_fn = CrossEntropyLoss()
        return loss_fn(predicted, target)


class MNISTViewer(Viewer):
    def view(self, batch, output):
        x,y = batch
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        rand_indices = [random.randint(0, x.size(0)-1) for _ in range(5)]
        for i,j in enumerate(rand_indices):
            img_tensor = x[j].squeeze()
            img = img_tensor.cpu().numpy()
            predicted_label = output[j].argmax().item()
            actual_label = y[j].item()
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(f"Actual: {actual_label}, Predicted: {predicted_label}")
            axs[i].axis('off')
        plt.show()

    def save(self, output: Tensor):
        pass
