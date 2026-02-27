"""
Training utilities.
Example submission.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import sys
import os
import glob
import re

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import cross_entropy, gradient_clipping


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    batch_size: int = 8
    log_interval: int = 50
    checkpoint_dir: Optional[str] = None  # save path (Google Drive)
    filename_prefix: str = "model"        # prefix (like "pretrain", "finetune")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, compute_loss_fn: Optional[Callable] = None):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Scheduler setup
        total_steps = len(train_dataloader) * config.num_epochs
        if config.warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
            main = CosineAnnealingLR(self.optimizer, T_max=total_steps - config.warmup_steps)
            self.scheduler = SequentialLR(self.optimizer, [warmup, main], milestones=[config.warmup_steps])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
            
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

    def _default_lm_loss(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.config.device)
        labels = batch["labels"].to(self.config.device)
        logits = model(input_ids)
        batch_size, seq_len, vocab_size = logits.shape
        return cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
    
    def _save_checkpoint(self, epoch: int, val_loss: float = None):
        """Save model checkpoint to disk."""
        if not self.config.checkpoint_dir:
            return
            
        # Create directory if not exists
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Filename: e.g., pretrain_epoch_0.pt
        filename = f"{self.config.filename_prefix}_epoch_{epoch}.pt"
        save_path = os.path.join(self.config.checkpoint_dir, filename)
        
        # Save minimal state (model weights)
        # For full resume capability (optimizer state etc.), you'd save more, 
        # but for this assignment, saving weights is sufficient.
        torch.save(self.model.state_dict(), save_path)
        print(f"  [Checkpoint] Saved model to {save_path}")

    def _load_latest_checkpoint(self) -> int:
        """
        Look for existing checkpoints and load the latest one.
        Returns the next epoch index (0 if no checkpoint found).
        """
        if not self.config.checkpoint_dir:
            return 0
            
        # Search for files like "pretrain_epoch_*.pt"
        pattern = os.path.join(self.config.checkpoint_dir, f"{self.config.filename_prefix}_epoch_*.pt")
        files = glob.glob(pattern)
        
        if not files:
            return 0
            
        # Find the file with the highest epoch number
        # Extract numbers using regex
        latest_epoch = -1
        latest_file = None
        
        for f in files:
            match = re.search(r"epoch_(\d+).pt", f)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_file = f
        
        if latest_file:
            print(f"\n[Resuming] Found checkpoint: {latest_file}")
            print(f"Loading weights and resuming from Epoch {latest_epoch + 1 + 1}...")
            # Load weights
            state_dict = torch.load(latest_file, map_location=self.config.device)
            self.model.load_state_dict(state_dict, strict=False) # strict=False allows flexibility
            return latest_epoch + 1 # Start from next epoch
            
        return 0

    def train_epoch(self, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_batches = len(self.train_dataloader)
        
        start_time = time.time()
        
        for batch in self.train_dataloader:
            self.optimizer.zero_grad()
            loss = self.compute_loss_fn(batch, self.model)
            loss.backward()
            gradient_clipping(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if self.global_step % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                steps_per_sec = num_batches / elapsed if elapsed > 0 else 0
                
                print(f"  [Epoch {epoch_idx+1}] Step {num_batches}/{total_batches} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Speed: {steps_per_sec:.2f} it/s")

        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_dataloader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_dataloader:
            loss = self.compute_loss_fn(batch, self.model)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self) -> Dict[str, Any]:
        # [Resume Logic] Try to find the latest checkpoint
        start_epoch = self._load_latest_checkpoint()
        
        if start_epoch >= self.config.num_epochs:
            print(f"Training already completed ({start_epoch} / {self.config.num_epochs} epochs).")
            return {"train_losses": [], "val_losses": []}
            
        print(f"Starting/Resuming training from Epoch {start_epoch + 1} to {self.config.num_epochs}...")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            print(f"--> Epoch {epoch + 1} finished. Avg Train Loss: {train_loss:.4f}")
            
            val_loss = None
            if self.val_dataloader:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                print(f"--> Validation Loss: {val_loss:.4f}")
            
            # [Save Logic] Save checkpoint at the end of EVERY epoch
            self._save_checkpoint(epoch, val_loss)
                
        return {"train_losses": self.train_losses, "val_losses": self.val_losses}


def compute_qa_loss(batch: Dict[str, torch.Tensor], model: nn.Module, device: str = "cuda") -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids, attention_mask)
    return cross_entropy(logits, labels)


def create_qa_loss_fn(device: str = "cuda") -> Callable:
    return lambda batch, model: compute_qa_loss(batch, model, device)