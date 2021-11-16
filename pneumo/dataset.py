import glob
import os
from typing import Optional

import albumentations as alb
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset
from torch.utils.data import Dataset

from pneumo.mask_functions import mask2rle, rle2mask
import torch
from typing import Dict, Any

class PneumoDataset(Dataset):
    def __init__(
        self,
        masks_path,
        dcm_paths,
        transforms: Optional[alb.core.composition.Compose] = None,
    ):
        self.df = pd.read_csv(masks_path)
        self.fns = dcm_paths
        self.transforms = transforms

    def load_dcm(self, dcm: FileDataset):
        return {
            "img": dcm.pixel_array / 255,
            "age": dcm.PatientAge,
            "sex": dcm.PatientSex,
            "view_position": dcm.ViewPosition,
        }

    def get_mask(self, uid: str, height: int, width: int):
        """Grabs masks and merges multiple masks into one 2d grid"""
        rle_masks = self.df[self.df["ImageId"] == uid]["EncodedPixels"].values

        if len(rle_masks) > 0 and rle_masks[0] != "-1":
            mask_arrs = np.clip(
                np.sum(
                    np.stack(
                        [rle2mask(m, height, width).T for m in rle_masks], axis=-1
                    ),
                    axis=-1,
                ),
                0,
                1,
            )
        else:
            mask_arrs = np.zeros((height, width))
        return mask_arrs

    def __getitem__(self, idx: int):
        """Return a dict to be used by the data loader"""
        fn = self.fns[idx]
        dcm = pydicom.dcmread(fn)
        dcm_data = self.load_dcm(dcm)
        dcm_data["target"] = self.get_mask(
            os.path.splitext(os.path.basename(fn))[0],
            dcm_data["img"].shape[0],
            dcm_data["img"].shape[1],
        )

        if self.transforms:
            transformed = self.transforms(
                image=dcm_data["img"], mask=dcm_data["target"]
            )
            dcm_data["img"] = transformed["image"]
            dcm_data["target"] = transformed["mask"]
        dcm_data["img"] = np.expand_dims(dcm_data["img"], axis=0).astype(np.float32)
        dcm_data["target"] = dcm_data["target"].astype(np.float32)
        return self.to_tensors(dcm_data)

    def to_tensors(self, data: Dict[str, Any]):
        return {
            "img": torch.as_tensor(data["img"]).cuda().half(),
            "target": torch.as_tensor(data["target"]).cuda()
        }

    def __len__(self):
        return len(self.fns)

class PneumoDataset(Dataset):
    def __init__(
        self,
        masks_path,
        dcm_paths,
        transforms: Optional[alb.core.composition.Compose] = None,
    ):
        self.df = pd.read_csv(masks_path)
        self.fns = dcm_paths
        self.transforms = transforms

    def load_dcm(self, dcm: FileDataset):
        return {
            "img": dcm.pixel_array / 255,
            "age": dcm.PatientAge,
            "sex": dcm.PatientSex,
            "view_position": dcm.ViewPosition,
        }

    def get_mask(self, uid: str, height: int, width: int):
        """Grabs masks and merges multiple masks into one 2d grid"""
        rle_masks = self.df[self.df["ImageId"] == uid]["EncodedPixels"].values

        if len(rle_masks) > 0 and rle_masks[0] != "-1":
            mask_arrs = np.clip(
                np.sum(
                    np.stack(
                        [rle2mask(m, height, width).T for m in rle_masks], axis=-1
                    ),
                    axis=-1,
                ),
                0,
                1,
            )
        else:
            mask_arrs = np.zeros((height, width))
        return mask_arrs

    def __getitem__(self, idx: int):
        """Return a dict to be used by the data loader"""
        fn = self.fns[idx]
        dcm = pydicom.dcmread(fn)
        dcm_data = self.load_dcm(dcm)
        dcm_data["target"] = self.get_mask(
            os.path.splitext(os.path.basename(fn))[0],
            dcm_data["img"].shape[0],
            dcm_data["img"].shape[1],
        )

        if self.transforms:
            transformed = self.transforms(
                image=dcm_data["img"], mask=dcm_data["target"]
            )
            dcm_data["img"] = transformed["image"]
            dcm_data["target"] = transformed["mask"]
        dcm_data["img"] = np.expand_dims(dcm_data["img"], axis=0).astype(np.float32)
        dcm_data["target"] = dcm_data["target"].astype(np.float32)
        return self.to_tensors(dcm_data)

    def to_tensors(self, data: Dict[str, Any]):
        return {
            "img": torch.as_tensor(data["img"]).cuda().half(),
            "target": torch.as_tensor(data["target"]).cuda()
        }

    def __len__(self):
        return len(self.fns)