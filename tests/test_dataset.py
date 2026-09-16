"""Dataset preparation tests without network access or real dataset files."""

from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

from dlbench.a1.data.dataset import prepare_data


class DatasetPreparationTests(unittest.TestCase):
    def test_prepare_data_forwards_download_flag_to_both_splits(self):
        config = {
            "data": {
                "root": "data",
                "split_file": "unused.json",
                "split_seed": 36,
                "validation_size": 10_000,
                "download": False,
            },
            "training": {"batch_size": 32},
            "run": {"seed": 69420},
        }
        fake_dataset = type("FakeDataset", (), {"targets": [], "classes": []})()

        with tempfile.TemporaryDirectory() as temp_dir:
            config["data"]["split_file"] = str(Path(temp_dir) / "split.json")
            manifest = type(
                "FakeManifest",
                (),
                {
                    "split_seed": 36,
                    "train_indices": [],
                    "validation_indices": list(range(10_000)),
                    "test_indices": [],
                },
            )()
            with patch("dlbench.a1.data.dataset.load_official_dataset", return_value=fake_dataset) as load_dataset, \
                 patch("dlbench.a1.data.dataset.create_split", return_value=manifest), \
                 patch("dlbench.a1.data.dataset.save_split"), \
                 patch("dlbench.a1.data.dataset.compute_normalization", return_value=([0.5], [0.25])):
                prepare_data(config)

        self.assertEqual(load_dataset.call_count, 2)
        self.assertEqual(
            [call.kwargs["download"] for call in load_dataset.call_args_list],
            [False, False],
        )


if __name__ == "__main__":
    unittest.main()