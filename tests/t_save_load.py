import uuid
from holytools.devtools import Unittest
from tests.mnist import MnistMLP
from torch.utils.data import Dataset

# ---------------------------------------------------------

class TestThunderModel(Unittest):

    def test_thunder_model_round_trip(self):
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0  # No data

            def __getitem__(self, idx):
                raise IndexError("Empty dataset")


        fpath = f'/tmp/py_thunder_/test_{uuid.uuid4()}.ckpt'
        empty_dataloader = EmptyDataset()
        original = MnistMLP()
        original.train_on(train_data=empty_dataloader)
        original.save(fpath)

        new = MnistMLP.load(fpath=fpath)

        reloadable_keys = ['_trainer', '_hparams','optimizer']
        original_attr = {k: v for k, v in original.__dict__.items() if k not in reloadable_keys}
        new_attr = {k: v for k, v in new.__dict__.items() if k not in reloadable_keys}

        print(f'Original state dict: {original_attr}')
        print(f'New state dict     : {new_attr}')
        self.assertEqual(str(original_attr),str(new_attr))

if __name__ == "__main__":
    TestThunderModel.execute_all()