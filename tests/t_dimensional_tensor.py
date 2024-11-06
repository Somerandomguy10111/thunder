from thunder.dtypes import Tensor1D, Tensor2D, Tensor3D, Tensor4D
from holytools.devtools import Unittest
import torch

class TestDimensionalTensors(Unittest):

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            _ = Tensor1D(torch.randn(2, 3))
        with self.assertRaises(ValueError):
            _ = Tensor2D(torch.randn(2))
        with self.assertRaises(ValueError):
            _ = Tensor3D(torch.randn(2, 3, 4, 5))
        with self.assertRaises(ValueError):
            _ = Tensor4D(torch.randn(2))


if __name__ == "__main__":
    TestDimensionalTensors.execute_all()