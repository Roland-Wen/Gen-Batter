import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging

from src.const import *


def pad_truncate(arr: np.ndarray, seq_len: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad or truncate the first dimension of `arr` to length `seq_len`.

    - If arr.shape[0] > seq_len, returns arr[:seq_len, ...].
    - If arr.shape[0] < seq_len, appends rows filled with `pad_value`.
    - Works for 1D arrays (shape (N,)) and ND arrays (shape (N, D1, D2, ...)).

    Parameters:
    -----------
    arr : np.ndarray
        Input array whose leading dimension is the sequence axis.
    seq_len : int
        Desired length along the first dimension.
    pad_value : float, optional (default=0.0)
        Value to use for padding.

    Returns:
    --------
    np.ndarray
        Array of shape (seq_len, ...) where "..." matches arr.shape[1:].
    """
    arr = np.asarray(arr)
    length = arr.shape[0]

    if length >= seq_len:
        # Simply truncate
        return arr[:seq_len, ...]

    # Need to pad
    pad_shape = (seq_len - length, *arr.shape[1:])
    pad_block = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([arr, pad_block], axis=0)


class AtBatSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, 
                 num_feats: list = NUMERICAL_FEATS_BASE, cat_feats: list = CATEGORICAL_FEATS_BASE, 
                 target_action_col: str = TARGET_ACTION_COL,
                 target_ev_col: str = 'target_ev', target_la_col: str = 'target_la',
                 seq_len: int = 10, 
                 scaler: StandardScaler = None, fit_scaler: bool = True):
        """
        df:                pandas DataFrame of processed data.
        num_feats:         list of column names for numerical inputs.
        cat_feats:         list of column names for categorical inputs (already int codes).
        target_action_col: name of the integer class label column.
        target_ev_col:     name of the continuous launch_speed target.
        target_la_col:     name of the continuous launch_angle target.
        seq_len:           maximum pitch sequence length
        scaler:            if provided, a fitted StandardScaler to normalize numericals.
        fit_scaler:        if True, fit a new StandardScaler on df[numerical_features].
        """
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.target_action_col = target_action_col
        self.target_ev_col = target_ev_col
        self.target_la_col = target_la_col

        # ----- scaler for numerical features -----
        if fit_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(self.df[self.num_feats].values)
            logging.info("Fitted scaler")
        else:
            if scaler is None:
                raise ValueError("Must provide a scaler if fit_scaler=False")
            self.scaler = scaler

        # Group by AB id
        grouped = df.groupby('at_bat_id').indices
        self.atbats = list(grouped.keys())
        self.ab_groups = grouped
        
        logging.info(f"{len(self.atbats)} ABs created")
        
    def __len__(self):
        return len(self.atbats)

    def __getitem__(self, idx):
        '''
        returns (
          'numerical': FloatTensor of shape (seq_len, num_numerical),
          'categorical': [LongTensor(seq_len) for each cat feature],
          'action': LongTensor(seq_len),
          'cont': FloatTensor(seq_len, 2),
          'mask': BoolTensor(seq_len)   #  True for real pitches, False for padding
        )
        '''
        ab_idxs = self.ab_groups[self.atbats[idx]]
        ab_pitches = self.df.iloc[ab_idxs].sort_values('pitch_number')
        # logging.info(f"Returning AB id {self.atbats[idx]}")
        
        # extract and pad/truncate numeric
        num = ab_pitches[self.num_feats].values.astype(np.float32)
        num = pad_truncate(num, self.seq_len)            # (seq_len, num_feats)
        
        # likewise for each cat feature
        cats = []
        for col in self.cat_feats:
            arr = ab_pitches[col].values.astype(np.int64)
            cats.append(pad_truncate(arr, self.seq_len))  # each is (seq_len,)

        # targets
        actions = pad_truncate(ab_pitches[self.target_action_col].values, self.seq_len)
        cont = pad_truncate(ab_pitches[[self.target_ev_col, self.target_la_col]].values.astype(np.float32), self.seq_len)

        # mask: 1 for real rows, 0 for pad
        mask = np.array([1]*min(len(ab_pitches), self.seq_len) + [0]*max(0, self.seq_len-len(ab_pitches)), dtype=bool)

        # convert to tensors
        return (
          torch.from_numpy(num),
          *[torch.from_numpy(c) for c in cats],
          torch.from_numpy(actions),
          torch.from_numpy(cont),
          torch.from_numpy(mask)
        )
