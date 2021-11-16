"""Originally had this written for another """

import gc
import os
import pickle
import traceback
from collections import defaultdict
from time import time

import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

import logging

def train_model(
    model,
    train_dataset,
    valid_dataset,
    batch_sz=8,
    initial_lr=0.001,
    weight_decay=0.05,
    max_epochs=1024,
    resume_from_epoch=None,
    num_workers=0,
    log_metrics=True,
    model_name="",
    project_name="",
    fp16=True,
):
    """
    Training loop for pytorch model. Wandb
    is used for logging and is required to be installed and init'd (wand.init(name='name',
    project='project') if watch_model and log_weights=True.

    Args:
        model (pytorch nn.Module): The model to train
        train_dataset (pytorch.utils.Dataset): Pytorch training dataset
        valid_dataset (pytorch.utils.Dataset): Pytorch validation dataset
        batch_sz (int): Batch size to use for dataloader
        initial_lr (float) : learning rate to use with AdamW optimiser
        max_epochs (int): epochs to run for
        resume_from_epoch (int): Continue training from a certain epoch. Used for logging purposes.
        num_workers (int): number of processes to use with dataloader
        watch_model (bool): whether to watch the model in wandb
        log_weights (bool): whether to log the model metrics into wandb
        model_name (str): model name to save the best to

    """

    device = torch.device("cuda")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=initial_lr, weight_decay=weight_decay
    )
    history = {
        "train": {
            "loss": [],
        },
        "valid": {
            "loss": [],
        },
        "epoch_time": [],
        "batch_time": [],
    }

    train_start_time = time()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_sz,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    torch.cuda.empty_cache()

    fp16_scaler = torch.cuda.amp.GradScaler() if fp16 else None

    logging.info(
        f"Training with {len(train_dataset)} and validation {len(valid_dataset)} for {max_epochs} epochs"
    )

    for epoch in range(max_epochs):
        if resume_from_epoch and epoch < resume_from_epoch:
            continue

        logging.info(f"Epoch {epoch}")
        epoch_start = time()
        train_epoch_metrics = train_epoch(model, train_loader, optimizer, fp16_scaler)
        valid_epoch_metrics = test_epoch(model, valid_loader)

        logging.info(train_epoch_metrics)
        logging.info(valid_epoch_metrics)
        logging.info(f"epoch_time {time() - epoch_start}")
        logging.info(f"batch_time {train_epoch_metrics['batch_time']}")

        for k in history["train"].keys():
            history["train"][k].append(train_epoch_metrics[k])

        for k in history["valid"].keys():
            history["valid"][k].append(valid_epoch_metrics[k])

        history["epoch_time"].append(time() - epoch_start)
        history["batch_time"].append(train_epoch_metrics["batch_time"])

        # Pre-pend with train or valid to it.
        if log_metrics:
            import wandb

            wandb.log(
                {"_".join(["train", k]): v[-1] for k, v in history["train"].items()},
                step=epoch,
            )
            wandb.log(
                {"_".join(["valid", k]): v[-1] for k, v in history["valid"].items()},
                step=epoch,
            )
            wandb.log({"epoch_time": history["epoch_time"][-1]}, step=epoch)
            wandb.log({"batch_time": history["batch_time"][-1]}, step=epoch)
            wandb.log({"time_elapsed": float(time() - train_start_time) / 60})

        save_best_model(model, history, os.path.join(project_name, model_name + ".pkl"))

    return model, history


def save_best_model(model, history, save_path):

    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))

    latest = history["valid"]["label_pixel_acc"][-1]

    if len(history["epoch_time"]) > 1:
        if latest > np.max(history["valid"]["label_pixel_acc"][:-1]):
            with open(save_path, "wb") as f:
                pickle.dump(model, f)


def train_epoch(model, train_loader, optimizer, fp16_scaler):
    model.train()

    batch_metrics = defaultdict(list)
    batch_time = time()

    for batch in tqdm(train_loader):
        forward_batch = {
            "mask": batch["mask"].cuda(),
            "image": batch["image"].cuda()
            if batch.get("image", None) is not None
            else None,
        }

        optimizer.zero_grad()

        if fp16_scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model.forward_batch(forward_batch)
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        else:
            loss = model.forward_batch(forward_batch)
            loss.backward()
            optimizer.step()

        output = model.predict(forward_batch)
        acc_metrics = model.get_batch_accuracies(output, forward_batch)

        # Append batch metrics
        for k, v in acc_metrics.items():
            batch_metrics[k].append(v)
        batch_metrics["loss"].append(loss.item())
        batch_metrics["batch_time"].append(float(time() - batch_time))
        batch_time = time()

    # Get epoch metrics
    epoch_metrics = {k: np.mean(v) for k, v in batch_metrics.items()}
    return epoch_metrics


def test_epoch(model, test_loader):
    model.eval()
    batch_metrics = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            forward_batch = {
                "mask": batch["mask"].cuda(),
                "image": batch["image"].cuda()
                if batch.get("image", None) is not None
                else None,
            }

            loss = model.forward_batch(forward_batch)
            output = model.predict(forward_batch)
            try:
                acc_metrics = model.get_batch_accuracies(output, forward_batch)
            except ZeroDivisionError as e:
                traceback.print_exc()
                continue
            # Append batch metrics
            for k, v in acc_metrics.items():
                batch_metrics[k].append(v)
                batch_metrics["loss"].append(loss.item())

    epoch_metrics = {k: np.mean(v) for k, v in batch_metrics.items()}
    return epoch_metrics
