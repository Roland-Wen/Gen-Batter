import pybaseball
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Callable, Optional
from sklearn.preprocessing import StandardScaler # Or other scaler
from collections import defaultdict
from tqdm.notebook import tqdm

# --- 3. PyTorch Dataset ---
class PitchSequenceDataset(Dataset):
    """
    PyTorch Dataset to handle pitch sequences for training the Transformer model.
    Groups data by at-bat, creates sequences, applies scaling, handles padding,
    and returns features, targets, and masks.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 numerical_features: List[str],
                 categorical_features: List[str],
                 target_action_col: str,
                 target_ev_la_cols: List[str], # Should be ['target_ev', 'target_la']
                 seq_length: int,
                 cat_mappings: Optional[Dict[str, Dict[str, int]]] = None, # Pre-computed mappings
                 cat_vocab_sizes: Optional[Dict[str, int]] = None, # Required if cat_mappings provided
                 scaler: Optional[StandardScaler] = None, # Pre-fitted scaler
                 device: torch.device = torch.device('cpu')):
        """
        Initializes the dataset, processes data into sequences, fits/applies scaler
        and categorical mappings if not provided.
        """
        super().__init__() # Important for inheritance

        if data.empty:
            logging.error("PitchSequenceDataset received an empty DataFrame. Cannot initialize.")
            # Handle appropriately - perhaps raise error or set internal state indicating emptiness
            self.sequences = []
            self._scaler = None
            self._cat_mappings = {}
            self._cat_vocab_sizes = {}
            self.device = device
            return

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_action_col = target_action_col
        self.target_ev_la_cols = target_ev_la_cols
        self.seq_length = seq_length
        self.device = device

        # Ensure targets exist
        if target_action_col not in data.columns:
             raise ValueError(f"Target action column '{target_action_col}' not found in data.")
        for col in target_ev_la_cols:
             if col not in data.columns:
                 raise ValueError(f"Target continuous column '{col}' not found in data.")


        # 1. Handle Scaler
        if scaler is None:
            logging.info("Fitting StandardScaler on numerical features...")
            self._scaler = StandardScaler()
            # Ensure numerical features exist before fitting
            present_num_features = [f for f in self.numerical_features if f in data.columns]
            if not present_num_features:
                 logging.warning("No numerical features found to fit scaler.")
                 self._scaler = None # Cannot fit
            else:
                 # TODO: shouldn't there be one scaler per feature?
                 self._scaler.fit(data[present_num_features])
        else:
            logging.info("Using pre-fitted StandardScaler.")
            self._scaler = scaler

        # Apply Scaler (if fitted)
        if self._scaler:
            present_num_features = [f for f in self.numerical_features if f in self._scaler.feature_names_in_]
            data[present_num_features] = self._scaler.transform(data[present_num_features])
            logging.info("Applied StandardScaler.")


        # 2. Handle Categorical Mappings
        self._cat_mappings = {}
        self._cat_vocab_sizes = {}
        if cat_mappings is None:
            logging.info("Creating categorical mappings...")
            self._cat_mappings = {}
            self._cat_vocab_sizes = {}
            for col in self.categorical_features:
                 if col in data.columns:
                    # Ensure column is treated as string/category type
                    unique_vals = data[col].astype(str).unique()
                    # Assign 0 to padding/unknown? No, let's map existing values from 0 upwards
                    mapping = {val: i for i, val in enumerate(unique_vals)}
                    self._cat_mappings[col] = mapping
                    # Vocab size includes all unique values found
                    self._cat_vocab_sizes[col] = len(mapping)
                    logging.info(f"  Mapped '{col}' ({len(mapping)} unique values).")
                 else:
                     logging.warning(f"Categorical feature '{col}' not found in data. Skipping mapping.")

        else:
            logging.info("Using pre-defined categorical mappings.")
            self._cat_mappings = cat_mappings
            # Ensure vocab sizes match provided mappings if they were also provided
            if cat_vocab_sizes:
                 self._cat_vocab_sizes = cat_vocab_sizes
            else: # Infer from provided mappings
                 self._cat_vocab_sizes = {col: len(mapping) for col, mapping in cat_mappings.items()}


        # Apply Categorical Mappings
        self.transformed_cat_data = {}
        for col, mapping in self._cat_mappings.items():
            if col in data.columns:
               # Use mapping, handle unseen values (map to a default like -1 or a specific index if needed)
               # For simplicity, assume all values were seen during mapping creation or are handled by 'Unknown' category
               data[col+'_mapped'] = data[col].astype(str).map(mapping).fillna(-1).astype(int) # Use -1 for unknown/unseen? Check embedding layer
               # Let's map N/A or unseen to 0 for now, assuming embedding layer has padding_idx=0
               # data[col+'_mapped'] = data[col].astype(str).map(mapping).fillna(0).astype(int)
               logging.info(f"Applied mapping for '{col}'.")
               self.transformed_cat_data[col] = data[col+'_mapped'].values # Store as numpy for faster access
            else:
                 logging.warning(f"Categorical feature '{col}' not found during mapping application.")


        # Store transformed numerical data and targets as numpy arrays for efficiency
        self.transformed_num_data = data[self.numerical_features].values if self._scaler else None
        self.target_action = data[self.target_action_col].values
        self.target_ev_la = data[self.target_ev_la_cols].values


        # 3. Group by At-Bat and Create Sequence Indices
        logging.info("Grouping data by at-bat and creating sequence indices...")
        self.sequences = [] # List to store (group_idx, end_row_idx_in_data) tuples
        # Use 'at_bat_id' created in preprocessing
        if 'at_bat_id' not in data.columns:
             raise ValueError("'at_bat_id' column not found. Preprocessing might have failed.")

        # Get indices corresponding to each group for faster lookup
        grouped = data.groupby('at_bat_id')
        self.group_indices = {name: group.index.tolist() for name, group in grouped}
        self.at_bat_ids = list(self.group_indices.keys())

        total_pitches = 0
        for at_bat_id in self.at_bat_ids:
            indices_in_df = self.group_indices[at_bat_id]
            num_pitches_in_at_bat = len(indices_in_df)
            total_pitches += num_pitches_in_at_bat
            # For each pitch in the at-bat, create a sequence ending at that pitch
            for i in range(num_pitches_in_at_bat):
                end_row_idx = indices_in_df[i]
                # Store the group identifier and the index *in the original dataframe*
                self.sequences.append((at_bat_id, end_row_idx))

        if not self.sequences:
             logging.warning("No sequences were generated from the provided data.")
        else:
             logging.info(f"Created {len(self.sequences)} sequences from {len(self.at_bat_ids)} at-bats ({total_pitches} total pitches).")


    def __len__(self) -> int:
        """Returns the total number of sequences (samples) in the dataset."""
        return len(self.sequences)

    # Modify PitchSequenceDataset.__getitem__

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves the idx-th sample sequence, target action, target EV/LA,
        and padding mask as tensors.
        """
        # --- Start Debug ---
        if idx >= len(self.sequences):
             # This check should ideally not be needed if DataLoader works correctly, but belt-and-suspenders
             logging.error(f"Requested index {idx} is >= dataset length {len(self.sequences)}")
             raise IndexError(f"Requested index {idx} is out of bounds for dataset length {len(self.sequences)}")
        # --- End Debug ---

        at_bat_id, end_row_idx = self.sequences[idx]

        # --- Start Debug ---
        if at_bat_id not in self.group_indices:
             logging.error(f"at_bat_id '{at_bat_id}' from sequences not found in group_indices!")
             # This would indicate a major issue during initialization
             # Handle appropriately, maybe return dummy data or raise error
             raise KeyError(f"at_bat_id '{at_bat_id}' not found in group indices.")
        # --- End Debug ---

        indices_in_df = self.group_indices[at_bat_id]

        try: # Add try block around index calculation
            end_idx_in_group = indices_in_df.index(end_row_idx)
        except ValueError:
            logging.error(f"end_row_idx {end_row_idx} not found within its supposed group list (at_bat_id={at_bat_id})!")
            logging.error(f"Group list (first 10): {indices_in_df[:10]}")
            raise # Re-raise the error, something is wrong with stored indices

        start_idx_in_group = max(0, end_idx_in_group - self.seq_length + 1)
        sequence_row_indices = indices_in_df[start_idx_in_group : end_idx_in_group + 1]
        actual_seq_len = len(sequence_row_indices)

        # --- DEBUG: Check indices before accessing numpy array ---
        if not sequence_row_indices:
             logging.warning(f"Warning in __getitem__(idx={idx}): sequence_row_indices is empty.")
             # Handle this case, maybe return zero tensors? For now, let it proceed to potentially fail later if needed.

        max_req_index = -1
        min_req_index = -1
        if sequence_row_indices: # Check if list is not empty
            max_req_index = max(sequence_row_indices)
            min_req_index = min(sequence_row_indices)

        current_data_shape_num = self.transformed_num_data.shape[0] if self.transformed_num_data is not None else -1
        current_data_shape_act = self.target_action.shape[0]

        if max_req_index >= current_data_shape_num or min_req_index < 0:
             logging.error(f"!!! Potential IndexError in __getitem__(idx={idx}) !!!")
             logging.error(f"  Indices Range: Min={min_req_index}, Max={max_req_index}")
             logging.error(f"  Num Array Shape[0]: {current_data_shape_num}")
             logging.error(f"  Act Array Shape[0]: {current_data_shape_act}") # Check target array size too
             logging.error(f"  Originating from at_bat_id={at_bat_id}, end_row_idx={end_row_idx}")
             logging.error(f"  Calculated start_idx_in_group: {start_idx_in_group}, end_idx_in_group: {end_idx_in_group}")
             logging.error(f"  sequence_row_indices (first 5): {sequence_row_indices[:5]}")
             # Raise a more informative error *before* the NumPy access fails
             raise IndexError(f"Index issue detected: Max index {max_req_index} or Min index {min_req_index} invalid for array size {current_data_shape_num}.")

        # --- END DEBUG ---


        # --- Extract Features (Original code causing the error) ---
        # Numerical features
        if self.transformed_num_data is not None:
             # This is the line that likely errored
             num_features_seq = self.transformed_num_data[sequence_row_indices]
             num_feat_dim = num_features_seq.shape[1]
        else:
             num_features_seq = np.empty((actual_seq_len, 0))
             num_feat_dim = 0

        # ...(Rest of the __getitem__ code remains the same)...
        # Categorical features
        cat_feature_seqs = []
        cat_feat_dim_total = 0
        for col in self.categorical_features:
            if col in self.transformed_cat_data:
                cat_seq = self.transformed_cat_data[col][sequence_row_indices]
                cat_feature_seqs.append(cat_seq[:, np.newaxis]) # Add axis for concatenation
                cat_feat_dim_total += 1
            else:
                 pass

        # Concatenate features
        if cat_feature_seqs:
            cat_features_seq_combined = np.concatenate(cat_feature_seqs, axis=1)
            features_seq_np = np.concatenate((num_features_seq, cat_features_seq_combined), axis=1).astype(np.float32)
        elif num_feat_dim > 0:
            features_seq_np = num_features_seq.astype(np.float32)
        else:
             total_features_dim = len(self.numerical_features) + len(self.categorical_features)
             features_seq_np = np.zeros((actual_seq_len, total_features_dim), dtype=np.float32)


        # --- Extract Targets (for the *last* pitch in sequence) ---
        target_action_val = self.target_action[end_row_idx]
        target_ev_la_val = self.target_ev_la[end_row_idx].astype(np.float64)

        # --- Padding ---
        padding_len = self.seq_length - actual_seq_len
        padding_mask = torch.zeros(self.seq_length, dtype=torch.bool) # False = Real data
        if padding_len < 0:
             # This should not happen if seq_length logic is correct, but add check
             logging.error(f"Negative padding length calculated in __getitem__(idx={idx})! actual_seq_len={actual_seq_len}, seq_length={self.seq_length}")
             padding_len = 0 # Avoid error, but indicates logic issue
             # Potentially truncate features_seq_np if it's too long? Or error out.
             # features_seq_np = features_seq_np[-self.seq_length:] # Keep only last seq_length elements


        if padding_len > 0:
            # Pad features at the beginning
            pad_value = 0.0
            feature_padding = np.full((padding_len, features_seq_np.shape[1]), pad_value, dtype=features_seq_np.dtype)
            features_seq_np = np.concatenate((feature_padding, features_seq_np), axis=0)
            # Mark padding positions as True in the mask
            padding_mask[:padding_len] = True
        elif features_seq_np.shape[0] > self.seq_length:
            # Added check: Handle cases where sequence might be longer than seq_length (shouldn't happen with current logic)
             logging.warning(f"Sequence length ({features_seq_np.shape[0]}) exceeded target seq_length ({self.seq_length}) for idx={idx}. Truncating.")
             features_seq_np = features_seq_np[-self.seq_length:]


        # Final shape check before converting to tensor
        if features_seq_np.shape[0] != self.seq_length:
             logging.error(f"Final features_seq_np shape[0] ({features_seq_np.shape[0]}) != seq_length ({self.seq_length}) for idx={idx}")
             # Handle this error - maybe return dummy data or raise
             raise ValueError(f"Incorrect final sequence length for idx={idx}")


        # --- Convert to Tensors ---
        features_tensor = torch.tensor(features_seq_np, dtype=torch.float32).to(self.device) # Model likely expects float
        action_tensor = torch.tensor(target_action_val, dtype=torch.long).to(self.device)
        ev_la_tensor = torch.tensor(target_ev_la_val, dtype=torch.float32).to(self.device)
        padding_mask_tensor = padding_mask.to(self.device) # Bool tensor

        return {
            'features': features_tensor,         # Shape: [seq_length, num_total_features]
            'action': action_tensor,             # Shape: [] (scalar)
            'ev_la': ev_la_tensor,               # Shape: [2]
            'padding_mask': padding_mask_tensor  # Shape: [seq_length], True where padded
        }

    def get_scaler(self) -> Optional[StandardScaler]:
        """Returns the fitted scaler."""
        return self._scaler

    def get_cat_mappings(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Returns the fitted categorical mappings."""
        return self._cat_mappings

    def get_cat_vocab_sizes(self) -> Optional[Dict[str, int]]:
         """Returns the vocabulary sizes for categorical features."""
         return self._cat_vocab_sizes

    def get_feature_indices(self) -> Dict[str, List[int]]:
        """Returns indices corresponding to numerical and categorical features in the final tensor."""
        num_indices = list(range(len(self.numerical_features)))
        cat_start_index = len(self.numerical_features)
        cat_indices = list(range(cat_start_index, cat_start_index + len(self.categorical_features)))
        return {"numerical": num_indices, "categorical": cat_indices}


# --- 4. DataLoader Creation ---
def create_dataloaders(train_data: pd.DataFrame,
                       val_data: pd.DataFrame, # Add test_data if needed
                       numerical_features: List[str],
                       categorical_features: List[str],
                       target_action_col: str,
                       target_ev_la_cols: List[str],
                       seq_length: int,
                       batch_size: int,
                       device: torch.device) -> Tuple[DataLoader, DataLoader, Optional[StandardScaler], Optional[Dict], Optional[Dict]]:
    """
    Creates training and validation DataLoaders.
    Resets index on data splits before creating Datasets. # <--- Added explanation
    """
    logging.info("Resetting index on training and validation data splits...")
    # --- Reset index here ---
    # Make copies to avoid SettingWithCopyWarning if these are slices
    train_data = train_data.copy().reset_index(drop=True)
    val_data = val_data.copy().reset_index(drop=True)
    # ----------------------------

    logging.info("Creating Training Dataset...")
    train_dataset = PitchSequenceDataset(
        data=train_data, # Pass the re-indexed data
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_action_col=target_action_col,
        target_ev_la_cols=target_ev_la_cols,
        seq_length=seq_length,
        scaler=None, # Fit on train data
        cat_mappings=None, # Fit on train data
        device=device
    )

    # Get fitted scaler and mappings from train_dataset to apply to val_dataset
    fitted_scaler = train_dataset.get_scaler()
    fitted_cat_mappings = train_dataset.get_cat_mappings()
    fitted_cat_vocab_sizes = train_dataset.get_cat_vocab_sizes() # Get vocab sizes too

    logging.info("Creating Validation Dataset...")
    # Ensure validation data isn't empty before creating dataset
    if val_data.empty:
        logging.warning("Validation data split is empty. Creating an empty DataLoader.")
        val_loader = None # Or handle as appropriate downstream
    else:
        val_dataset = PitchSequenceDataset(
            data=val_data, # Pass the re-indexed data
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_action_col=target_action_col,
            target_ev_la_cols=target_ev_la_cols,
            seq_length=seq_length,
            scaler=fitted_scaler, # Use scaler fitted on train
            cat_mappings=fitted_cat_mappings, # Use mappings from train
            cat_vocab_sizes=fitted_cat_vocab_sizes, # Use vocab sizes from train
            device=device
        )

    # Create DataLoaders
    #num_workers = 4 if device.type == 'cuda' else 0
    #pin_memory = True if device.type == 'cuda' else False
    num_workers = 0
    pin_memory = False

    logging.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    # Only create val_loader if val_dataset exists
    if not val_data.empty:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2, # Often use larger batch size for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
         val_loader = None # Explicitly set to None if no validation data


    logging.info("DataLoaders created.")
    return train_loader, val_loader, fitted_scaler, fitted_cat_mappings, fitted_cat_vocab_sizes