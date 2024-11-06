import uuid
from holytools.devtools import Unittest
from tests.mnist import MnistMLP

# ---------------------------------------------------------

class TestThunderSaveLoad(Unittest):
    def test_thunder_round_trip(self):
        fpath = f'/tmp/py_thunder_/test_{uuid.uuid4()}.ckpt'
        original = MnistMLP()
        original.save(fpath)

        new = MnistMLP.load(fpath=fpath)

        for p1, p2 in zip(original.parameters(), new.parameters()):
            params_unequal = p1.data.ne(p2.data).sum() > 0
            print(p1.data, p2.data)
            self.assertTrue(not params_unequal)

        c1 = original.compute_configs
        c2 = new.compute_configs
        print(f'Origianl compute configs: {c1}')
        print(f'New compute configs: {c2}')
        self.assertEqual(c1.device, c2.device)


if __name__ == "__main__":
    TestThunderSaveLoad.execute_all()