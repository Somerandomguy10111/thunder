from thunder.dtypes import Tensor1D, Tensor2D, Tensor3D, Tensor4D
from hollarek.devtools import Unittest
import torch

class TestDimensionalTensors(Unittest):

    def test_tensor_dimensions(self):
        tensor_4d = Tensor4D(torch.randn(2, 3, 4, 5))
        tensor_3d = tensor_4d[0]  # Should return Tensor3D
        tensor_2d = tensor_3d[0]  # Should return Tensor2D
        tensor_1d = tensor_2d[0]  # Should return Tensor1D
        scalar = tensor_1d[0]

        self.assertIsInstance(tensor_3d, Tensor3D, "Indexing Tensor4D did not return Tensor3D")
        self.assertIsInstance(tensor_2d, Tensor2D, "Indexing Tensor3D did not return Tensor2D")
        self.assertIsInstance(tensor_1d, Tensor1D, "Indexing Tensor2D did not return Tensor1D")
        self.assertIsInstance(scalar, torch.Tensor, "Indexing Tensor1D did not return a scalar tensor")

        self.assertEqual(scalar.dim(), 0, "The returned scalar is not 0-dimensional")

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