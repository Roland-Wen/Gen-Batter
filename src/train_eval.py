import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import math
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Callable, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit # For splitting data by game
import numpy as np
from tqdm.notebook import tqdm # For progress bars

# --- 5. Training Loop ---
def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                loss_fn: Callable,
                optimizer: optim.Optimizer,
                device: torch.device,
                bip_action_index: int) -> float:
    """
    Trains the model for one epoch.

    Returns:
        Average loss for the epoch.
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_batches = len(dataloader)

    # Wrap dataloader with tqdm for progress bar
    pbar = tqdm(dataloader, total=num_batches, desc="Training Epoch")

    for batch in pbar:
        # Move batch to device
        features = batch['features'].to(device)
        true_action = batch['action'].to(device)
        true_ev_la = batch['ev_la'].to(device)
        # Ensure padding mask is boolean
        padding_mask = batch['padding_mask'].to(device, dtype=torch.bool)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, src_padding_mask=padding_mask)

        # Calculate loss
        loss = loss_fn(outputs, true_action, true_ev_la, bip_action_index)

        # Backward pass and optimization
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item()) # Update progress bar

    return total_loss / num_batches


# --- 6. Evaluation Loop ---
def evaluate_model(model: nn.Module,
                   dataloader: DataLoader,
                   loss_fn: Callable,
                   device: torch.device,
                   bip_action_index: int) -> Dict[str, float]:
    """
    Evaluates the model on a given dataset (validation or test).

    Returns:
        Dictionary containing evaluation metrics (e.g., 'loss', 'accuracy', 'mae_ev', 'mae_la').
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    correct_actions = 0
    total_abs_error_ev = 0.0
    total_abs_error_la = 0.0
    bip_samples_count = 0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, total=num_batches, desc="Evaluating")

    with torch.no_grad():  # Disable gradient calculations
        for batch in pbar:
            # Move batch to device
            features = batch['features'].to(device)
            true_action = batch['action'].to(device)
            true_ev_la = batch['ev_la'].to(device)
            padding_mask = batch['padding_mask'].to(device, dtype=torch.bool)
            batch_size = features.size(0)
            total_samples += batch_size

            # Forward pass
            outputs = model(features, src_padding_mask=padding_mask)

            # Calculate loss
            loss = loss_fn(outputs, true_action, true_ev_la, bip_action_index)
            total_loss += loss.item() * batch_size # Accumulate total loss, not average

            # Calculate action accuracy
            pred_action_indices = torch.argmax(outputs['action_logits'], dim=1)
            correct_actions += (pred_action_indices == true_action).sum().item()

            # Calculate conditional MAE for EV/LA
            is_bip_true_mask = (true_action == bip_action_index)
            bip_indices = torch.where(is_bip_true_mask)[0]
            num_bip_in_batch = len(bip_indices)

            if num_bip_in_batch > 0:
                bip_samples_count += num_bip_in_batch
                true_ev_la_bip = true_ev_la[bip_indices]

                # Get point estimate from MDN (mean of most likely component)
                mdn_alpha_logits_bip = outputs['mdn_alpha_logits'][bip_indices]
                mdn_mu_bip = outputs['mdn_mu'][bip_indices]
                # mdn_sigma_bip = outputs['mdn_sigma'][bip_indices] # Not needed for mean

                most_likely_component_idx = torch.argmax(mdn_alpha_logits_bip, dim=1) # [num_bip_in_batch]

                # Gather the mu values corresponding to the most likely component
                # Need to use gather or advanced indexing
                pred_ev_la_bip = mdn_mu_bip[torch.arange(num_bip_in_batch), most_likely_component_idx, :] # [num_bip_in_batch, 2]

                # Calculate absolute errors
                abs_errors = torch.abs(pred_ev_la_bip - true_ev_la_bip)
                total_abs_error_ev += abs_errors[:, 0].sum().item()
                total_abs_error_la += abs_errors[:, 1].sum().item()

    # Calculate final metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_actions / total_samples if total_samples > 0 else 0.0
    mae_ev = total_abs_error_ev / bip_samples_count if bip_samples_count > 0 else 0.0
    mae_la = total_abs_error_la / bip_samples_count if bip_samples_count > 0 else 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mae_ev_bip': mae_ev,
        'mae_la_bip': mae_la,
        'bip_samples_evaluated': bip_samples_count
    }
