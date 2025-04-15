import pybaseball
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import math
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Callable, Optional
from sklearn.preprocessing import StandardScaler # Or other scaler
from collections import defaultdict
from tqdm.notebook import tqdm

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50): # max_len >= SEQ_LENGTH
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- Transformer Model ---
class BatterActionTransformer(nn.Module):
    def __init__(self,
                 # Feature dimensions / indices
                 num_numerical_features: int,
                 cat_feature_indices: List[int], # Indices of categorical features in input tensor
                 cat_vocab_sizes: Dict[str, int], # Vocab size for each cat feature (ensure order matches indices)
                 # Model hyperparameters
                 d_model, nhead, num_encoder_layers, dim_feedforward,
                 # Output dimensions
                 num_actions, num_mdn_components,
                 # Other params
                 seq_length=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.num_mdn_components = num_mdn_components
        self.output_dim_continuous = 2 # EV, LA
        self.cat_feature_indices = cat_feature_indices # Store indices

        # Calculate total dimension for categorical embeddings
        # Assuming each categorical feature gets its own embedding layer mapped to d_model/N space
        # Simpler: embed each to a smaller dimension then concat, or embed all to d_model and sum/concat
        # Let's embed each to d_model and sum them with numerical projection
        self.embedding_layers = nn.ModuleDict()
        cat_feature_names = list(cat_vocab_sizes.keys()) # Get names in order
        if len(cat_feature_names) != len(cat_feature_indices):
             logging.warning("Mismatch between cat_vocab_sizes keys and cat_feature_indices length. Ensure order is consistent.")
        for i, idx in enumerate(cat_feature_indices):
             # Need a way to link index back to feature name for vocab size lookup
             # Assume the order of cat_feature_indices matches the keys in cat_vocab_sizes
             feature_name = cat_feature_names[i]
             vocab_size = cat_vocab_sizes[feature_name]
             # Use padding_idx=0 if 0 was used for padding/unknown in Dataset
             self.embedding_layers[feature_name] = nn.Embedding(vocab_size, d_model, padding_idx=None) # Set padding_idx if needed

        # Linear projection for numerical features
        self.numerical_proj = nn.Linear(num_numerical_features, d_model) if num_numerical_features > 0 else None

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Output layers (same as before)
        self.action_head = nn.Linear(d_model, num_actions)
        self.mdn_alpha_head = nn.Linear(d_model, num_mdn_components)
        self.mdn_mu_head = nn.Linear(d_model, num_mdn_components * self.output_dim_continuous)
        self.mdn_sigma_head = nn.Linear(d_model, num_mdn_components * self.output_dim_continuous)
        self.softplus = nn.Softplus()

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_padding_mask=None):
        """
        Args:
            src: Input tensor, shape [batch_size, seq_len, num_total_features]
                 Features are expected to be ordered: [numerical..., categorical...]
            src_padding_mask: Bool tensor, shape [batch_size, seq_len], True indicates padding
        """
        batch_size, seq_len, _ = src.shape
        device = src.device

        # --- Input Processing ---
        # Separate numerical and categorical features based on indices
        numerical_src = src[:, :, :self.numerical_proj.in_features] if self.numerical_proj else None
        categorical_src = src[:, :, self.cat_feature_indices] # Shape [batch_size, seq_len, num_cat_features]

        # Project numerical features
        projected_numerical = self.numerical_proj(numerical_src) * math.sqrt(self.d_model) if self.numerical_proj else 0

        # Embed categorical features and sum them
        embedded_categorical = 0
        cat_feature_names = list(self.embedding_layers.keys())
        for i, feature_name in enumerate(cat_feature_names):
            # Get the correct slice for this categorical feature (needs long type for embedding)
            cat_input = categorical_src[:, :, i].long()
            # Apply embedding - handle potential padding_idx if set
            embedded_categorical += self.embedding_layers[feature_name](cat_input)

        # Combine numerical and categorical representations (summation)
        combined_embedding = projected_numerical + embedded_categorical # Shape [batch_size, seq_len, d_model]

        # Add positional encoding
        src_pos = self.pos_encoder(combined_embedding.permute(1, 0, 2)).permute(1, 0, 2) # Add pos encoding

        # --- Transformer Encoder ---
        src_mask_causal = self._generate_square_subsequent_mask(seq_len, device)
        # Transformer expects padding mask where True means IGNORE
        # Our dataset mask has True for padding, which matches src_key_padding_mask
        memory = self.transformer_encoder(src_pos, mask=src_mask_causal, src_key_padding_mask=src_padding_mask)

        # Use output of the last *non-padded* token if padding exists?
        # Simplified: use the output corresponding to the last position index (-1)
        # This assumes sequences are aligned such that the target corresponds to the last element.
        last_token_output = memory[:, -1, :] # Shape [batch_size, d_model]

        # --- Output Heads ---
        action_logits = self.action_head(last_token_output)
        mdn_alpha_logits = self.mdn_alpha_head(last_token_output)
        mdn_mu = self.mdn_mu_head(last_token_output).view(-1, self.num_mdn_components, self.output_dim_continuous)
        # Ensure sigma is positive and stable
        mdn_sigma = self.softplus(self.mdn_sigma_head(last_token_output)).view(-1, self.num_mdn_components, self.output_dim_continuous)
        mdn_sigma = torch.clamp(mdn_sigma, min=1e-4) # Add stability clamp

        return {
            'action_logits': action_logits,
            'mdn_alpha_logits': mdn_alpha_logits,
            'mdn_mu': mdn_mu,
            'mdn_sigma': mdn_sigma
        }

# --- Custom Loss Function ---
def mdn_loss_fn(y_pred, y_true_action, y_true_ev_la, bip_action_index=3):
    """
    Calculates combined CCE and MDN NLL loss.

    Args:
        y_pred (dict): Model outputs {'action_logits':..., 'mdn_alpha_logits':...}
        y_true_action (Tensor): True action indices [batch_size]
        y_true_ev_la (Tensor): True EV/LA values [batch_size, 2] (NaNs/zeros if not BIP)
        bip_action_index (int): Index corresponding to BallInPlay action.
    """
    action_logits = y_pred['action_logits']
    mdn_alpha_logits = y_pred['mdn_alpha_logits']
    mdn_mu = y_pred['mdn_mu']
    mdn_sigma = y_pred['mdn_sigma']

    # Action Loss (Categorical Cross Entropy) - Applies to all samples
    action_loss = nn.functional.cross_entropy(action_logits, y_true_action, reduction='none')

    # MDN Loss - Only for BIP samples
    is_bip_true = (y_true_action == bip_action_index)
    bip_indices = torch.where(is_bip_true)[0]

    mdn_nll = torch.zeros_like(action_loss) # Initialize MDN loss per sample as zero

    if len(bip_indices) > 0:
        # Select parameters and true values for BIP samples
        mdn_alpha_logits_bip = mdn_alpha_logits[bip_indices]
        mdn_mu_bip = mdn_mu[bip_indices]
        mdn_sigma_bip = mdn_sigma[bip_indices]
        y_true_ev_la_bip = y_true_ev_la[bip_indices]

        # Ensure sigmas are not too small
        min_sigma = 1e-4
        mdn_sigma_bip = torch.clamp(mdn_sigma_bip, min=min_sigma)

        # Create the GMM distribution
        mix = D.Categorical(logits=mdn_alpha_logits_bip)
        # Using Normal for independent EV/LA components for simplicity
        comp = D.Independent(D.Normal(loc=mdn_mu_bip, scale=mdn_sigma_bip), 1) # Reinterpret batch dim as event dim
        gmm = D.MixtureSameFamily(mix, comp)

        # Calculate negative log likelihood for true EV/LA
        log_prob_bip = gmm.log_prob(y_true_ev_la_bip)
        nll_bip = -log_prob_bip

        # Place NLL into the correct indices of the per-sample loss tensor
        # Using index_put_ for in-place update might be efficient
        mdn_nll.index_put_((bip_indices,), nll_bip)


    # Combined loss per sample
    total_loss_per_sample = action_loss + mdn_nll

    # Return average loss over the batch
    return total_loss_per_sample.mean()