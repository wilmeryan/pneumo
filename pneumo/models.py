import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pneumo import losses

# class UNetModel(nn.Module):
#     """Token grid embedding using UNet encoder/decoder."""

#     def __init__(self, unet_depth=3):
#         """
#         Uses https://github.com/qubvel/segmentation_models.pytorch

#         Args:
#             unet_depth (int, optional): depth of the Unet model. Defaults to 3.
#         """
#         super().__init__()
#         self.unet_depth = unet_depth

#         # 32 is the last channel size it goes down to. This would be [128, 64, 32] for depth=3
#         self.decoder_channels = [32 * (unet_depth - d + 1) for d in range(unet_depth)]

#         unet_model = smp.Unet(
#             encoder_name="efficientnet-b3",
#             encoder_depth=unet_depth,
#             decoder_channels=self.decoder_channels,
#             in_channels=3
#         )
#         self.encoder = unet_model.encoder
#         self.decoder = unet_model.decoder
#         self.segmentation_joiner = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(64, self.num_output_channels, 3, padding=1),
#             nn.BatchNorm2d(self.num_output_channels),
#         )
    
    # def forward(self, ):

class UNetLightning(pl.LightningModule):

    def __init__(self, unet_depth:int=3, learning_rate=0.1, smoothing=0.2):
        """
        Uses https://github.com/qubvel/segmentation_models.pytorch

        Args:
            unet_depth (int, optional): depth of the Unet model. Defaults to 3.
        """
        super().__init__()
        self.unet_depth = unet_depth

        # 32 is the last channel size it goes down to. This would be [128, 64, 32] for depth=3
        self.decoder_channels = [32 * (unet_depth - d + 1) for d in range(unet_depth)]

        unet_model = smp.Unet(
            encoder_name="efficientnet-b3",
            in_channels=1
        )
        self.encoder = unet_model.encoder
        self.decoder = unet_model.decoder
        self.segmentation_head = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 4, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 1, kernel_size=3)
        )
        self.losses = {
            "bce": nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(300)), # heuristic calculated
            "dice": losses.dice_loss, # functional,
            "focal": losses.focal_loss
        }
        self.smoothing = smoothing

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(*out)
        out = self.segmentation_head(out)
        return F.sigmoid(out)

    def training_step(self, batch, batch_idx):
        b_imgs, b_targets = batch["img"], batch["target"]
        if self.smoothing:
            b_targets = ((torch.ones(b_targets.size()) - self.smoothing) * b_targets) + self.smoothing/2
        
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.segmentation_head(out)

        focal_loss = self.losses["focal"](x, b_targets)
        bce_loss = self.losses["bce"](out, b_targets)
        dice_loss = self.losses["dice"](x, b_targets)
        loss = focal_loss + 3 * dice_loss + bce_loss

        self.log("train_focal_loss", focal_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters)