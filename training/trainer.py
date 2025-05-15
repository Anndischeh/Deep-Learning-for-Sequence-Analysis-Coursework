import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import os
from utils.helpers import save_model # Assuming helper function exists
import wandb
import config



class ModelTrainer:
    """Handles the training and validation loops."""

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, model_save_path, is_bert=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_save_path = model_save_path # Path to save the best model
        self.is_bert = is_bert # Flag to handle different input formats
        self.best_valid_loss = float('inf') # Initialize best validation loss


    def _train_epoch(self):
        """Performs one epoch of training."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()

            if self.is_bert:
                # BERT input is a dictionary
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device).unsqueeze(1) # Ensure labels are [batch_size, 1]
                outputs = self.model(input_ids, attention_mask)
            else:
                # RNN/CNN input is (sequences, labels)
                sequences, labels = batch
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1) # Ensure labels are [batch_size, 1]
                outputs = self.model(sequences)

            loss = self.criterion(outputs, labels)
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            with torch.no_grad():
                preds = torch.sigmoid(outputs) >= 0.5 # Apply sigmoid and threshold
                correct = (preds == labels.byte()).sum().item() # Use .byte() for comparison if labels are float
                total_correct += correct
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def _evaluate_epoch(self):
        """Performs one epoch of validation."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if self.is_bert:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device).unsqueeze(1)
                    outputs = self.model(input_ids, attention_mask)
                else:
                    sequences, labels = batch
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device).unsqueeze(1)
                    outputs = self.model(sequences)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy
                preds = torch.sigmoid(outputs) >= 0.5
                correct = (preds == labels.byte()).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def train(self, n_epochs):
        """Runs the full training loop for n_epochs, logging metrics to W&B."""
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        print(f"Starting training for {n_epochs} epochs on {self.device}...")
        if config.LOG_WANDB and wandb.run:
            # Watch model gradients (optional, can be computationally expensive)
            # wandb.watch(self.model, log_freq=100) # log='all' or log='gradients' or log='parameters'
            pass # Placeholder if not watching immediately

        for epoch in range(n_epochs):
            start_time = time.time()

            train_loss, train_acc = self._train_epoch()
            valid_loss, valid_acc = self._evaluate_epoch()

            end_time = time.time()
            epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

            # Store history locally
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(valid_loss)
            history['val_acc'].append(valid_acc)

            # Log metrics to W&B (if enabled)
            if config.LOG_WANDB and wandb.run:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": valid_loss,
                    "val/accuracy": valid_acc,
                    "epoch_duration_secs": end_time - start_time
                }
                # Check if best model and log that too
                if valid_loss < self.best_valid_loss:
                     log_dict["val/best_loss"] = valid_loss # Log the best loss achieved so far

                try:
                    wandb.log(log_dict) # Log all metrics for the epoch
                except Exception as e:
                    print(f"Warning: Could not log epoch metrics to W&B: {e}")


            # Save the best model based on validation loss (local saving)
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
                save_model(self.model, self.model_save_path)
                print(f"\tBest model saved to {self.model_save_path} (Val Loss: {valid_loss:.4f})")
                # Optional: Save model as W&B artifact
                if config.LOG_WANDB and wandb.run:
                   try:
                       model_artifact = wandb.Artifact(f'{wandb.run.name}-model', type='model')
                       model_artifact.add_file(self.model_save_path)
                       wandb.log_artifact(model_artifact, aliases=['best']) # Mark as best
                       print(f"\tBest model also saved as W&B artifact.")
                   except Exception as e:
                       print(f"Warning: Could not save model artifact to W&B: {e}")


            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc*100:.2f}%')

        print("Training finished.")
        # Optional: Upload final model if not done during training
        if config.LOG_WANDB and wandb.run and os.path.exists(self.model_save_path):
           try:
               model_artifact = wandb.Artifact(f'{wandb.run.name}-model', type='model')
               model_artifact.add_file(self.model_save_path)
               wandb.log_artifact(model_artifact, aliases=['final']) # Mark as final
               print(f"Final model saved as W&B artifact.")
           except Exception as e:
               print(f"Warning: Could not save final model artifact to W&B: {e}")
        return history

    def _epoch_time(self, start_time, end_time):
        """Calculates the time taken for an epoch."""
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs