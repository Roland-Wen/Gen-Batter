import pybaseball
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import time
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Callable, Optional
from sklearn.preprocessing import StandardScaler # Or other scaler
from collections import defaultdict
from tqdm.notebook import tqdm
import warnings


CACHE_DIR = "dataset"

def fetch_statcast_data_for_year(year: int, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """
    Fetches pitch-by-pitch Statcast data for a given year.
    Checks for cached raw data first, unless it's the current year (always refetches).
    Saves fetched raw data to cache.
    """
    raw_cache_file = os.path.join(cache_dir, f"raw_statcast_{year}.parquet")
    current_year = datetime.now().year

    # --- Cache Check (Skip for current year) ---
    if year < current_year and os.path.exists(raw_cache_file):
        try:
            logging.info(f"Loading cached raw data for {year} from {raw_cache_file}")
            df = pd.read_parquet(raw_cache_file)
            logging.info(f"Loaded {len(df)} pitches from cache for {year}.")
            # Basic check if loaded data looks reasonable (e.g., has expected columns)
            if 'game_pk' in df.columns and 'pitch_type' in df.columns:
                 return df
            else:
                 logging.warning(f"Cached file {raw_cache_file} seems incomplete or corrupted. Refetching.")
        except Exception as e:
            logging.error(f"Error loading cached file {raw_cache_file}: {e}. Refetching.")

    # --- Fetching Logic (if no valid cache or current year) ---
    if year == current_year:
        logging.info(f"Fetching fresh data for current year {year} (cache ignored).")
    else:
        logging.info(f"No valid cache found for {year}. Fetching from pybaseball...")


    # Define date range
    start_date = date(year, 3, 1)
    end_date = date(year, 11, 30)
    today = date.today()

    if year > today.year:
        logging.warning(f"Year {year} is in the future. No data available.")
        return pd.DataFrame()
    if year == today.year and end_date >= today:
       end_date = today - timedelta(days=1)
       if end_date < start_date:
           logging.warning(f"Current date {today} is before fetch start date {start_date} for year {year}. No data available yet.")
           return pd.DataFrame()

    logging.info(f"Fetching Statcast data for {year} from {start_date} to {end_date}")
    all_data = []
    current_start = start_date
    fetch_interval_days = 7

    while current_start <= end_date:
        current_end = current_start + timedelta(days=fetch_interval_days - 1)
        if current_end > end_date:
            current_end = end_date
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        logging.info(f"  Fetching chunk: {start_str} to {end_str}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                data_chunk = pybaseball.statcast(start_dt=start_str, end_dt=end_str, verbose=False)
                if data_chunk is not None and not data_chunk.empty:
                    all_data.append(data_chunk)
                    logging.info(f"  Fetched {len(data_chunk)} pitches.")
        except ValueError:
             logging.warning(f"  Value error (likely no games) for {start_str} to {end_str}.")
        except Exception as e:
            logging.error(f"  Error fetching chunk {start_str} to {end_str}: {e}")
        current_start += timedelta(days=fetch_interval_days)
        time.sleep(2)

    if not all_data:
        logging.warning(f"No data fetched for year {year}.")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Finished fetching for {year}. Total raw pitches: {len(final_df)}")

    # --- Save to Cache ---
    try:
        logging.info(f"Saving raw data for {year} to {raw_cache_file}")
        final_df.to_parquet(raw_cache_file, index=False)
    except Exception as e:
        logging.error(f"Error saving raw data cache file {raw_cache_file}: {e}")

    return final_df


def preprocess_and_feature_engineer(raw_df: pd.DataFrame, year: int = -1, cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """
    Cleans raw Statcast data, engineers features (like target action),
    selects relevant columns, handles missing values, and identifies
    feature types (numerical, categorical). Excludes final pitch location.
    Adds unique at-bat identifiers if not present.
    """
    processed_cache_file = os.path.join(cache_dir, f"processed_statcast_{year}.parquet")
    current_year = datetime.now().year

    # --- Cache Check ---
    if os.path.exists(processed_cache_file):
        try:
            logging.info(f"Loading cached processed data for {year} from {processed_cache_file}")
            df = pd.read_parquet(processed_cache_file)
            logging.info(f"Loaded {len(df)} pitches from cache for {year}.")
            # Basic check if loaded data looks reasonable (e.g., has expected columns)
            if 'at_bat_id' in df.columns and 'target_action' in df.columns:
                 return df
            else:
                 logging.warning(f"Cached file {processed_cache_file} seems incomplete or corrupted.")
        except Exception as e:
            logging.error(f"Error loading cached file {processed_cache_file}: {e}. Reprocessing.")

    
    if raw_df.empty:
        logging.warning("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    logging.info(f"Preprocessing data. Initial shape: {raw_df.shape}")
    df = raw_df.copy()

    # Ensure necessary base columns exist
    # These are crucial for basic processing and target/feature creation
    base_cols = ['release_pos_x', 'release_pos_z', 'effective_speed', 'spin_axis', 'release_spin_rate', 'pfx_x', 'pfx_z',
                 'batter', 'pitcher', 'stand', 'p_throws',
                 'balls', 'strikes', 
                 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'bat_score', 'fld_score',
                 'inning', 'inning_topbot', 
                 'description', 'launch_speed', 'launch_angle', 
                 'game_pk', 'at_bat_number', 'pitch_number']
    missing_base = [col for col in base_cols if col not in df.columns]
    if missing_base:
        # Allow processing to continue but warn heavily, imputation might fail later
        logging.warning(f"Potentially critical base columns missing, results may be unreliable: {missing_base}")
        # Optional: Raise error if certain columns are absolutely essential
        # if 'description' not in df.columns or 'game_pk' not in df.columns:
        #     raise ValueError("Essential columns 'description' or 'game_pk' missing.")


    # 1. Filter irrelevant pitches (Keep only standard competitive pitch outcomes)
    # This list might need tuning based on data exploration
    valid_descriptions = [
        'called_strike', 'ball', 'swinging_strike',
        'swinging_strike_blocked', 'foul', 'foul_tip', 'blocked_ball',
        'hit_into_play'
        # Excludes pitchouts, intentional balls/walks, catcher interference etc.
    ]
    # Only filter if 'description' column exists
    if 'description' in df.columns:
       df = df[df['description'].isin(valid_descriptions)].copy()
       logging.info(f"Shape after filtering descriptions: {df.shape}")
       if df.empty:
           logging.warning("DataFrame empty after filtering descriptions.")
           return pd.DataFrame()
    else:
        logging.warning("Column 'description' not found, skipping description filtering.")


    # 2. Define Target Action (Numerical mapping: 0:Take, 1:S&M, 2:Foul, 3:BIP)
    action_map = {
        'ball': 0, 'called_strike': 0, 'blocked_ball': 0,
        'swinging_strike': 1, 'swinging_strike_blocked': 1,
        'foul': 2, 'foul_tip': 2,
        'hit_into_play': 3
    }

    if 'description' in df.columns:
        df['target_action'] = df['description'].map(action_map)
        # Drop rows where action couldn't be mapped (should only happen if filtering failed or map is incomplete)
        initial_len = len(df)
        df.dropna(subset=['target_action'], inplace=True)
        if len(df) < initial_len:
             logging.warning(f"Dropped {initial_len - len(df)} rows with unmappable descriptions for target_action.")
        if df.empty:
             logging.warning("DataFrame empty after mapping target action.")
             return pd.DataFrame()
        df['target_action'] = df['target_action'].astype(int)
    else:
        logging.error("Cannot create 'target_action' as 'description' column is missing.")
        return pd.DataFrame() # Cannot proceed without target


    # 3. Define Target Continuous (EV, LA) - Fill NaNs for non-BIP with 0
    bip_action_index = 3 # Defined in action_map
    # Only create targets if source columns exist
    if 'launch_speed' in df.columns:
        df['target_ev'] = df['launch_speed'].where(df['target_action'] == bip_action_index, 0.0)
        df.fillna({'target_ev': 0.0}, inplace=True) # Fill remaining NaNs (e.g., BIP w/ missing EV)
    else:
        logging.warning("Column 'launch_speed' not found. Creating 'target_ev' as all zeros.")
        df['target_ev'] = 0.0

    if 'launch_angle' in df.columns:
        df['target_la'] = df['launch_angle'].where(df['target_action'] == bip_action_index, 0.0)
        df.fillna({'target_la': 0.0}, inplace=True) # Fill remaining NaNs (e.g., BIP w/ missing LA)
    else:
        logging.warning("Column 'launch_angle' not found. Creating 'target_la' as all zeros.")
        df['target_la'] = 0.0
    logging.info("Target variables created/processed.")


    # 4. Create categorical feature mappings
    # pitcher / batter handeness
    df['stand'] = df['stand'].map({'L':0, 'R':1}).fillna(-1).astype(int)
    df['p_throws'] = df['p_throws'].map({'L':0, 'R':1}).fillna(-1).astype(int)
    
    # balls / strikes
    df['balls']   = df['balls'].astype(int)
    df['strikes'] = df['strikes'].astype(int)

    # game state
    df['on_1b'] = df['on_1b'].notna().astype(int)
    df['on_2b'] = df['on_1b'].notna().astype(int)
    df['on_3b'] = df['on_1b'].notna().astype(int)
    df['inning'] = df['inning'].astype(int)
    df['inning_topbot'] = df['inning_topbot'].map({'Top': 0, 'Bot': 1}).fillna(-1).astype(int)
    df['score_diff'] = df['bat_score'] - df['fld_score']

    # Define feature lists based on available columns and engineered features
    numerical_features_base = [
        'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z', 'release_spin_rate', 'spin_axis',
        'score_diff'
    ]
    runner_features = ['on_1b', 'on_2b', 'on_3b']
    categorical_features_base = [
        'batter', 'pitcher', # NOT categorized YET, need data from ALL YEARS
        'stand', 'p_throws',
        'balls', 'strikes', 'outs_when_up', 'inning', 'inning_topbot'
    ]

    # Filter lists based on actual columns present in df
    numerical_features = [f for f in numerical_features_base if f in df.columns] + runner_features
    categorical_features = [f for f in categorical_features_base if f in df.columns]

    identifier_cols = ['game_pk', 'at_bat_number', 'pitch_number'] # Keep for sorting/grouping
    target_cols = ['target_action', 'target_ev', 'target_la']

    # Ensure identifiers exist
    missing_ids = [col for col in identifier_cols if col not in df.columns]
    if missing_ids:
        raise ValueError(f"Essential identifier columns missing: {missing_ids}")


    # Keep only selected + identifier + target columns
    selected_cols = numerical_features + categorical_features + identifier_cols + target_cols
    # Ensure all selected cols exist before slicing (already filtered lists above)
    df = df[selected_cols].copy()
    logging.info(f"Shape after feature selection: {df.shape}")


    # 5. Handle Missing Values in Selected Predictor Features
    # Impute numerical features - using median for robustness
    for col in numerical_features:
        if df[col].isnull().any():
            median_val = df[col].median()
            df.fillna({col: median_val}, inplace=True)
            logging.info(f"Imputed NaNs in numerical feature '{col}' with median ({median_val:.2f}).")

    
    # 6. Create Unique At-Bat ID (robustly handles types)
    df['at_bat_id'] = df['game_pk'].astype(str) + '_' + df['at_bat_number'].astype(str)

    
    # 7. Sort data for sequencing (Crucial!)
    # Ensure pitch_number is numeric for correct sorting
    df['pitch_number'] = pd.to_numeric(df['pitch_number'], errors='coerce')
    df.dropna(subset=['pitch_number'], inplace=True) # Drop if pitch number is invalid
    df['pitch_number'] = df['pitch_number'].astype(int)

    df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'], inplace=True)
    df.reset_index(drop=True, inplace=True) # Good practice after sorting

    logging.info(f"Preprocessing complete. Final shape: {df.shape}")
    logging.info("Remember to categorize pitcher and batter!!!")
    # Log feature lists identified for clarity downstream
    logging.info(f"Final Numerical Features: {numerical_features}")
    logging.info(f"Final Categorical Features: {categorical_features}")

    try:
        logging.info(f"Saving processed data for {year} to {processed_cache_file}")
        df.to_parquet(processed_cache_file, index=False)
    except Exception as e:
        logging.error(f"Error saving processed data cache file {processed_cache_file}: {e}")
    
    return df # Remember to categorize pitcher and batter!!!

def categorize_pitcher_batter(df: pd.DataFrame):
    '''Categorize pitcher and batter, return cat codes'''
    cat_series = df['batter'].astype('category')
    df['batter'] = cat_series.cat.codes
    batter_categories = cat_series.cat.categories
    
    cat_series = df['pitcher'].astype('category')
    df['pitcher'] = cat_series.cat.codes
    pitcher_categories = cat_series.cat.categories

    return df, batter_categories, pitcher_categories