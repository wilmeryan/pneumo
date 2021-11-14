import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pneumo import losses


class UNetLightning(pl.LightningModule):

    def __init__(self, learning_rate=1e-2, smoothing=0.1):
        """
        Uses https://github.com/qubvel/segmentation_models.pytorch
        """
        super().__init__()

        unet_model = smp.Unet(
            encoder_name="efficientnet-b3",
            in_channels=1
        )
        self.encoder = unet_model.encoder
        self.decoder = unet_model.decoder
        self.segmentation_head = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 8, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 1, kernel_size=3, padding="same"),
            nn.Softmax(dim=1)
        )
        self.losses = {
            "bce": nn.BCELoss(),
            "dice": losses.dice_loss, # functional,
            "focal": losses.focal_loss
        }
        self.smoothing = smoothing
        self.learning_rate = learning_rate

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(*out)
        out = self.segmentation_head(out)
        return out

    def training_step(self, batch, batch_idx):
        b_imgs, b_targets = batch["img"], batch["target"].unsqueeze(1)
        if self.smoothing:
            b_targets = ((torch.ones(b_targets.size()).cuda() - self.smoothing) * b_targets) + self.smoothing/2
        
        out = self.encoder(b_imgs)
        out = self.decoder(*out)
        out = self.segmentation_head(out)

        focal_loss = self.losses["focal"](out, b_targets)
        bce_loss = self.losses["bce"](out, b_targets)
        dice_loss = self.losses["dice"](out, b_targets)
        loss = focal_loss + 3 * dice_loss + bce_loss

        self.log("train_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        b_imgs, b_targets = batch["img"], batch["target"].unsqueeze(1)

        out = self.encoder(b_imgs)
        out = self.decoder(*out)
        out = self.segmentation_head(out)

        focal_loss = self.losses["focal"](out, b_targets)
        bce_loss = self.losses["bce"](out, b_targets)
        dice_loss = self.losses["dice"](out, b_targets)
        loss = focal_loss + 3 * dice_loss + bce_loss

        self.log("valid_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     """Warm up from https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling"""
    #     # skip the first 500 steps
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr_scale * self.learning_rate

    #     # update params
    #     optimizer.step(closure=optimizer_closure)