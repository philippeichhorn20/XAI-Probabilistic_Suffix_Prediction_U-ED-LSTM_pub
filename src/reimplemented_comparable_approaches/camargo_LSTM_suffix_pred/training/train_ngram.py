"""Trainer for the Camargo n-gram pipeline.

Drops the `_preprocess_batch` logic of the original `Training` class because
the new `CamargoNGramDataset` yields `(prefix_cats, prefix_nums, target)`
tuples directly — no "last column of a pre-built window" reinterpretation.

Model `forward` returns logits (post the double-softmax fix); this trainer
feeds them straight into `F.cross_entropy`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class NGramTraining:
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 optimize_values,
                 writer=None,
                 save_model_n_th_epoch: int = 1,
                 saving_path: str = "reimpl_ngram_model.pkl",
                 num_workers: int = 0):
        self.device = device
        print("Device: ", device)
        self.data_train = data_train
        self.data_val = data_val
        self.model = model.to(self.device)

        self.optimizer = optimize_values["optimizer"]
        self.scheduler = optimize_values["scheduler"]
        self.epochs = optimize_values["epochs"]
        self.mini_batches = optimize_values["mini_batches"]
        self.shuffle = optimize_values["shuffle"]
        print(f"Optimizer: {self.optimizer}")
        print(f"Scheduler: {self.scheduler}")
        print(f"Epochs: {self.epochs}  Mini-batch: {self.mini_batches}  Shuffle: {self.shuffle}")

        self.writer = writer
        self.save_model_n_th_epoch = save_model_n_th_epoch
        self.saving_path = saving_path
        self.num_workers = num_workers

    def _to_device(self, cats_batch, nums_batch, target):
        cats_batch = [c.to(self.device) for c in cats_batch]
        nums_batch = [n.to(self.device) for n in nums_batch]
        target = target.long().to(self.device)
        return cats_batch, nums_batch, target

    def train(self):
        val_loader = DataLoader(
            self.data_val, batch_size=self.mini_batches, shuffle=False,
            num_workers=self.num_workers, pin_memory=False,
        )
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            loader = DataLoader(
                self.data_train, batch_size=self.mini_batches, shuffle=self.shuffle,
                num_workers=self.num_workers, pin_memory=False,
            )

            total, n_batches = 0.0, 0
            for cats_batch, nums_batch, target in loader:
                cats_batch, nums_batch, target = self._to_device(cats_batch, nums_batch, target)

                logits = self.model((cats_batch, nums_batch))
                loss = F.cross_entropy(logits, target)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total += loss.item()
                n_batches += 1

            epoch_loss = total / max(n_batches, 1)
            current_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            val_loss = self._validate(val_loader)

            tqdm.write(f"Epoch [{epoch + 1}/{self.epochs}], LR: {current_lr}")
            tqdm.write(f"Training:   Avg Loss: {epoch_loss:.4f}")
            tqdm.write(f"Validation: Avg Loss: {val_loss:.4f}")

            if self.writer is not None:
                self.writer.add_scalars("Hyperparameter", {"Learning Rate": current_lr}, epoch + 1)
                self.writer.add_scalars("Total Losses",
                                        {"Training Total": epoch_loss,
                                         "Validation Total": val_loss},
                                        epoch + 1)

            self.scheduler.step(val_loss)

            if self.save_model_n_th_epoch and (epoch + 1) % self.save_model_n_th_epoch == 0:
                tqdm.write("saving model")
                self.model.save(self.saving_path)

        print("Training complete.")
        self.model.save(self.saving_path)
        tqdm.write(f"Model saved to path: {self.saving_path}")

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total, n_batches = 0.0, 0
        for cats_batch, nums_batch, target in loader:
            cats_batch, nums_batch, target = self._to_device(cats_batch, nums_batch, target)
            logits = self.model((cats_batch, nums_batch))
            loss = F.cross_entropy(logits, target)
            total += loss.item()
            n_batches += 1
        return total / max(n_batches, 1)
