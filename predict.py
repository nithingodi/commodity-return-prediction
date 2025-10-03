# === IMPORTS ===
import os
import gc
import pickle
import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# === LOAD MODELS AND HISTORICAL DATA (GLOBAL SCOPE) ===
# This part runs only once when the notebook starts.

print("--- Initializing Models and Data ---")

# --- !! IMPORTANT !! ---
# Make sure this path matches the name you gave your uploaded dataset.
MODEL_DIR = '/kaggle/input/my-final-submission-models/' 
COMP_DIR = '/kaggle/input/mitsui-commodity-prediction-challenge/'

# Load the two trained models
with open(os.path.join(MODEL_DIR, 'storm_detector.pkl'), 'rb') as f:
    detector_model = pickle.load(f)
print("✅ Storm Detector model loaded.")

with open(os.path.join(MODEL_DIR, 'champion_models.pkl'), 'rb') as f:
    champion_models = pickle.load(f)
print("✅ Champion Models dictionary loaded.")

# Load the full training data to serve as the initial history for feature calculations
features_history_df = pd.read_csv(os.path.join(COMP_DIR, 'train.csv'), parse_dates=['date_id']).set_index('date_id')
print(f"Initial features history loaded with {len(features_history_df)} dates.")

# Load training labels to get the correct target column names in order
train_labels_df = pd.read_csv(os.path.join(COMP_DIR, 'train_labels.csv'))
TARGET_COLUMNS = [col for col in train_labels_df.columns if col.startswith('T')]
TARGET_NAME_MAP = {f'target_{i}': name for i, name in enumerate(TARGET_COLUMNS)}
REVERSE_TARGET_NAME_MAP = {name: f'target_{i}' for i, name in enumerate(TARGET_COLUMNS)}

# Initialize a dataframe to store the history of storm probabilities
storm_prob_history_df = pd.DataFrame(columns=['P(Storm_in_5_days)'], index=features_history_df.index).fillna(0)

# --- UPDATE ---
# Get the exact list of feature names the champion models expect
first_model_key = next(key for key, model in champion_models.items() if model is not None)
MODEL_COLUMNS = champion_models[first_model_key].feature_name_
print(f"✅ Model feature list loaded. Expecting {len(MODEL_COLUMNS)} features.")

print("✅ Initialization complete.")


# === KAGGLE API PREDICT FUNCTION ===
# This function will be called repeatedly by the evaluation server.

NUM_TARGET_COLUMNS = 424

def predict(
    test: pl.DataFrame,
    label_lags_1_batch: pl.DataFrame,
    label_lags_2_batch: pl.DataFrame,
    label_lags_3_batch: pl.DataFrame,
    label_lags_4_batch: pl.DataFrame,
) -> pl.DataFrame:
    """
    This function is called for each new date_id to generate predictions.
    """
    global features_history_df, storm_prob_history_df

    # --- 1. DATA PREPARATION AND STATE MANAGEMENT ---
    # --- UPDATE --- Removed 'use_pyarrow_numpy_strings' argument for compatibility
    test_pd = test.to_pandas().set_index('date_id')
    
    # Append new test data to our historical dataframe
    features_history_df = pd.concat([features_history_df, test_pd])
    
    # --- 2. STAGE 1: PRE-STORM DETECTOR ---
    precursor_features = pd.DataFrame(index=features_history_df.index)
    us_index_proxies = ['US_Stock_SPY_adj_close', 'US_SPX_Close', 'US_Stock_IVV_adj_close', 'US_Stock_VOO_adj_close', 'US_Stock_QQQ_adj_close']
    volatility_proxy = None
    for proxy in us_index_proxies:
        if proxy in features_history_df.columns:
            volatility_proxy = proxy
            break
    if not volatility_proxy and 'LME_CA_Close' in features_history_df.columns:
        volatility_proxy = 'LME_CA_Close'

    if volatility_proxy:
        vol_10d = features_history_df[volatility_proxy].pct_change().rolling(10).std()
        precursor_features['vol_of_vol_10d_on_60d'] = vol_10d.rolling(60).std()
    
    precursor_features = precursor_features.ffill().fillna(0)
    current_detector_features = precursor_features.loc[test_pd.index]
    storm_probability = detector_model.predict_proba(current_detector_features)[:, 1]
    
    current_storm_prob = pd.DataFrame(storm_probability, index=test_pd.index, columns=['P(Storm_in_5_days)'])
    storm_prob_history_df = pd.concat([storm_prob_history_df, current_storm_prob])

    # --- 3. STAGE 2: CHAMPION MODELS ---
    # --- UPDATE --- Start with a copy of the test data to include its base features
    df = test_pd.copy()

    # a) Create lagged target features from API data
    lag_dfs = {1: label_lags_1_batch, 2: label_lags_2_batch, 3: label_lags_3_batch}
    for lag, lag_df in lag_dfs.items():
        lag_pd = lag_df.to_pandas().rename(columns=TARGET_NAME_MAP)
        for col in TARGET_COLUMNS:
            df[f'{col}_lag_{lag}'] = lag_pd[col].values[0]

    # b) Create rolling features from the full history
    key_instruments = ['LME_CA_Close', 'LME_AH_Close', 'LME_ZS_Close', 'LME_PB_Close', 'US_Stock_SPY_adj_close', 'FX_EURUSD', 'JPX_Nikkei225_Close']
    existing_instruments = [inst for inst in key_instruments if inst in features_history_df.columns]
    
    for instrument in existing_instruments:
        for window in [5, 21]:
            roll_mean = features_history_df[instrument].rolling(window=window, min_periods=1).mean()
            roll_std = features_history_df[instrument].rolling(window=window, min_periods=1).std()
            df[f'{instrument}_roll_avg_{window}'] = roll_mean.loc[test_pd.index]
            df[f'{instrument}_roll_std_{window}'] = roll_std.loc[test_pd.index]
            
    # c) Add lagged storm probability
    df['P(Storm_in_5_days)_lag_1'] = storm_prob_history_df['P(Storm_in_5_days)'].shift(1).loc[test_pd.index]
    
    # d) Final cleaning and alignment
    X_test_final = df.ffill().fillna(0)
    # --- UPDATE --- Align columns to match the model's training data exactly
    X_test_aligned = X_test_final.reindex(columns=MODEL_COLUMNS, fill_value=0)

    # --- 4. GENERATE PREDICTIONS ---
    predictions = {}
    for target_col in champion_models.keys():
        model = champion_models[target_col]
        if model is not None:
            # Use the new, aligned dataframe for prediction
            pred = model.predict(X_test_aligned)[0]
            predictions[target_col] = pred
        else:
            predictions[target_col] = 0.0

    # --- 5. FORMAT FOR SUBMISSION ---
    predictions_pd = pd.DataFrame([predictions])
    predictions_pd = predictions_pd.rename(columns=REVERSE_TARGET_NAME_MAP)
    predictions_pd = predictions_pd[[f'target_{i}' for i in range(NUM_TARGET_COLUMNS)]]
    
    predictions_pl = pl.from_pandas(predictions_pd)
    
    gc.collect()
    
    return predictions_pl


# === SERVER SETUP (PROVIDED BY KAGGLE) ===
import kaggle_evaluation.mitsui_inference_server
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway((COMP_DIR,))
