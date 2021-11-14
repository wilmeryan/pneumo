import glob
import os
import unittest

import pandas as pd
from torch.utils.data import DataLoader

from pneumo.dataset import PneumoDataset


class TestDatset(unittest.TestCase):
    def setUp(self):

        # Fix this to use an environment var...
        project_path = "/home/wilmer-linux/Projects/data/pneumo_s1/dicom-images-train/"

        self.fns = glob.glob(os.path.join(project_path, "**/*.dcm"), recursive=True)
        self.label_path = os.path.join(project_path, "../train-rle.csv")
        self.ds = PneumoDataset(
            self.label_path, self.fns, None
        )

    def test_dataset(self):
        example = self.ds[0]
        self.assertEqual(example["img"].shape, (1, 1024, 1024))
        self.assertEqual(example["target"].shape, (1, 1024, 1024))

    def test_dataloader(self):
        dataloader = DataLoader(self.ds, batch_size=32, shuffle=True, num_workers=4)
        batch = next(iter(dataloader))
        self.assertEqual(batch["img"].shape, (32, 1, 1024, 1024))
        self.assertEqual(batch["target"].shape, (32, 1, 1024, 1024))
