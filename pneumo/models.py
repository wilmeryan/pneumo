import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pneumo import losses, metrics
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MetaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, age, sex, pos):
        # bs, 1
        meta = torch.stack([age, sex, pos], dim=-1)
        # bs ,3
        return self.encoder(meta)


class MetaUNetLightning(pl.LightningModule):

    def __init__(self, learning_rate=1e-3, smoothing=0.1):
        """
        Uses https://github.com/qubvel/segmentation_models.pytorch
        """
        super().__init__()

        unet_model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            in_channels=1
        )
        self.encoder = unet_model.encoder
        self.decoder = unet_model.decoder
        self.meta_encoder = MetaEncoder()
        self.seg_head = nn.Sequential(
            # nn.Dropout2d(0.1),
            nn.Conv2d(16, 4, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 1, kernel_size=3, padding="same"),
            nn.Sigmoid()
        )
        self.losses = {
            "bce": nn.BCELoss(),
            "dice": losses.dice_loss, # functional,
            "focal": losses.focal_loss,
            "wsdice": losses.weighted_soft_dice_loss
        }
        self.smoothing = smoothing
        self.learning_rate = learning_rate

    def forward(self, img, age, sex, pos):
        out = self.encoder(x)
        out = self.decoder(*out)
        out = self.seg_head(out)
        return out

    def training_step(self, batch, batch_idx):
        b_imgs, b_targets = batch["img"], batch["target"].unsqueeze(1).float()
        b_age, b_sex, b_pos = batch["age"], batch["sex"], batch["pos"]

        meta_bias = self.meta_encoder(b_age, b_sex, b_pos).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 256, 256)
        out = self.encoder(b_imgs + meta_bias)
        out = self.decoder(*out)
        out = self.seg_head(out).float()

        with autocast(enabled=False):
            dice_loss = self.losses["dice"](out, b_targets)
            ws_dice_loss = self.losses["wsdice"](out, b_targets)
            dice_coeff = metrics.dice_coeff(out, b_targets)

            if self.smoothing:
                b_targets = ((torch.ones(b_targets.size()).cuda() - self.smoothing) * b_targets) + self.smoothing/2

            focal_loss = self.losses["focal"](out, b_targets)
            # bce_loss = self.losses["bce"](out, b_targets)
            loss = focal_loss - torch.log(1 - dice_loss) #+ bce_loss

        self.log("train_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_wsdice_loss", ws_dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("train_bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice_coeff",dice_coeff, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        b_imgs, b_targets = batch["img"], batch["target"].unsqueeze(1).float()
        b_age, b_sex, b_pos = batch["age"], batch["sex"], batch["pos"]

        meta_bias = self.meta_encoder(b_age, b_sex, b_pos).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 256, 256)
        out = self.encoder(b_imgs + meta_bias)
        out = self.decoder(*out)
        out = self.seg_head(out).float()

        with autocast(enabled=False):
            dice_coeff = metrics.dice_coeff(out, b_targets)
            dice_loss = self.losses["dice"](out, b_targets)
            ws_dice_loss = self.losses["wsdice"](out, b_targets)

            focal_loss = self.losses["focal"](out, b_targets)
            # bce_loss = self.losses["bce"](out, b_targets)
            loss = focal_loss - torch.log(1 - dice_loss) # + bce_loss

        self.log("valid_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_wsdice_loss", ws_dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log("valid_bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_dice_coeff", dice_coeff, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer, patience=5), "monitor": "valid_loss_epoch"}
        return [optimizer], lr_schedulers


    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # skip the first 500 steps
        if self.trainer.global_step < 200:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 200.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

        # update params
        optimizer.step(closure=optimizer_closure)


class UNetLightning(pl.LightningModule):

    def __init__(self, learning_rate=1e-2, smoothing=0.1):
        """
        Uses https://github.com/qubvel/segmentation_models.pytorch
        """
        super().__init__()

        unet_model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights="imagenet",
            in_channels=1
        )
        self.encoder = unet_model.encoder
        self.decoder = unet_model.decoder
        self.seg_head = nn.Sequential(
            # nn.Dropout2d(0.1),
            nn.Conv2d(16, 4, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 1, kernel_size=3, padding="same"),
            nn.Sigmoid()
        )
        self.losses = {
            "bce": nn.BCELoss(),
            "dice": losses.dice_loss, # functional,
            "focal": losses.focal_loss,
            "wsdice": losses.weighted_soft_dice_loss
        }
        self.smoothing = smoothing
        self.learning_rate = learning_rate

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(*out)
        out = self.seg_head(out)
        return out

    def training_step(self, batch, batch_idx):
        b_imgs, b_targets = batch["img"], batch["target"].unsqueeze(1).float()
        
        out = self.encoder(b_imgs)
        out = self.decoder(*out)
        out = self.seg_head(out).float()

        with autocast(enabled=False):
            dice_loss = self.losses["dice"](out, b_targets)
            ws_dice_loss = self.losses["wsdice"](out, b_targets)
            dice_coeff = metrics.dice_coeff(out, b_targets)

            if self.smoothing:
                b_targets = ((torch.ones(b_targets.size()).cuda() - self.smoothing) * b_targets) + self.smoothing/2

            focal_loss = self.losses["focal"](out, b_targets)
            bce_loss = self.losses["bce"](out, b_targets)
            loss = focal_loss - torch.log(1 - dice_loss) # + bce_loss

        self.log("train_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_wsdice_loss", ws_dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice_coeff",dice_coeff, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        b_imgs, b_targets = batch["img"], batch["target"].unsqueeze(1).float()

        out = self.encoder(b_imgs)
        out = self.decoder(*out)
        out = self.seg_head(out).float()

        with autocast(enabled=False):
            dice_coeff = metrics.dice_coeff(out, b_targets)
            dice_loss = self.losses["dice"](out, b_targets)
            ws_dice_loss = self.losses["wsdice"](out, b_targets)

            focal_loss = self.losses["focal"](out, b_targets)
            bce_loss = self.losses["bce"](out, b_targets)
            loss = focal_loss - torch.log(1 - dice_loss) # + bce_loss

        self.log("valid_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_wsdice_loss", ws_dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_dice_coeff", dice_coeff, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer, patience=5), "monitor": "valid_loss_epoch"}
        return [optimizer], lr_schedulers


    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # skip the first 500 steps
        if self.trainer.global_step < 100:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 100.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

        # update params
        optimizer.step(closure=optimizer_closure)