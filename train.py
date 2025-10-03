import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import pickle

# ---
# HELPER FUNCTIONS
# ---
def get_asset_type(asset_name):
    if asset_name.startswith('LME_'): return 'Metal'
    if asset_name.startswith('JPX_'): return 'Future'
    if asset_name.startswith('US_Stock_'): return 'Stock'
    if asset_name.startswith('FX_'): return 'Currency'
    if asset_name.startswith('US_'): return 'Index'
    return 'Other'

def categorize_pair(pair_str):
    if isinstance(pair_str, str) and '-' in pair_str:
        type1, type2 = [get_asset_type(s.strip()) for s in pair_str.split('-')]
        return '-'.join(sorted([type1, type2]))
    elif isinstance(pair_str, str):
        return f'Single_{get_asset_type(pair_str)}'
    return 'Unknown'

def create_target_analysis_df(tain_labels_df: pd.DataFrame, target_pairs_df: pd.DataFrame):
    stats_df = pd.DataFrame({'volatility': tain_labels_df.std()}, index=tain_labels_df.columns).dropna()
    target_pairs_df['fundamental_category'] = target_pairs_df['pair'].apply(categorize_pair)
    analysis_df = stats_df.merge(target_pairs_df[['target', 'fundamental_category', 'pair']], left_index=True, right_on='target', how='left').set_index('target')
    return analysis_df

# ---
# PHASE 1: BUILD AND SAVE THE "PRE-STORM DETECTOR"
# ---
print("--- PHASE 1: Building and Saving the 'Pre-Storm Detector' ---")
train_df = pd.read_csv('train.csv').set_index('date_id')
tain_labels_df = pd.read_csv('train_labels.csv').set_index('date_id')
target_pairs_df = pd.read_csv('target_pairs.csv')
target_analysis = create_target_analysis_df(tain_labels_df, target_pairs_df)

global_avg_abs_return = tain_labels_df.abs().mean(axis=1)
storm_threshold = global_avg_abs_return.quantile(0.95)
storm_dates = global_avg_abs_return[global_avg_abs_return > storm_threshold].index
is_storm_day = pd.Series(0, index=train_df.index)
is_storm_day.loc[storm_dates] = 1
print(f"Identified {len(storm_dates)} global storm days.")

# --- Engineer precursor features with a robust fallback ---
precursor_features = pd.DataFrame(index=train_df.index)
us_index_proxies = ['US_Stock_SPY_adj_close', 'US_SPX_Close', 'US_Stock_IVV_adj_close', 'US_Stock_VOO_adj_close', 'US_Stock_QQQ_adj_close']
volatility_proxy = None
for proxy in us_index_proxies:
    if proxy in train_df.columns:
        volatility_proxy = proxy
        print(f"Found high-quality US index proxy: {volatility_proxy}")
        break
if not volatility_proxy and 'LME_CA_Close' in train_df.columns:
    volatility_proxy = 'LME_CA_Close'
    print(f"Warning: No US index proxy found. Falling back to {volatility_proxy}.")

if volatility_proxy:
    vol_10d = train_df[volatility_proxy].pct_change().rolling(10).std()
    precursor_features['vol_of_vol_10d_on_60d'] = vol_10d.rolling(60).std()
else:
     print("CRITICAL WARNING: No suitable proxy found for precursor features.")

precursor_features = precursor_features.ffill().fillna(0)
print("Created precursor features.")

y_detector = is_storm_day.shift(-5).fillna(0)
X_detector = precursor_features
detector_model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
detector_model.fit(X_detector, y_detector)
print("Successfully trained the Pre-Storm Detector model.")

with open('storm_detector.pkl', 'wb') as f:
    pickle.dump(detector_model, f)
print("✅ Saved storm_detector.pkl to disk.")

# ---
# PHASE 2: BUILD AND SAVE THE FINAL "CHAMPION" MODELS
# ---
print("\n--- PHASE 2: Building and Saving the Final Champion Models ---")
storm_probability = detector_model.predict_proba(X_detector)[:, 1]
storm_prob_series = pd.Series(storm_probability, index=X_detector.index, name='P(Storm_in_5_days)')

def feature_engineer_champion(features_df, targets_df, storm_signal):
    df = features_df.copy()
    for col in targets_df.columns:
        for lag in [1, 2, 3]: df[f'{col}_lag_{lag}'] = targets_df[col].shift(lag)
    key_instruments = ['LME_CA_Close', 'LME_AH_Close', 'LME_ZS_Close', 'LME_PB_Close', 'US_Stock_SPY_adj_close', 'FX_EURUSD', 'JPX_Nikkei225_Close']
    existing_instruments = [inst for inst in key_instruments if inst in df.columns]

    for instrument in existing_instruments:
        for window in [5, 21]:
            df[f'{instrument}_roll_avg_{window}'] = df[instrument].rolling(window=window, min_periods=1).mean()
            df[f'{instrument}_roll_std_{window}'] = df[instrument].rolling(window=window, min_periods=1).std()
    df['P(Storm_in_5_days)_lag_1'] = storm_signal.shift(1)
    return df

features_df_final = feature_engineer_champion(train_df, tain_labels_df, storm_prob_series)
features_df_final = features_df_final.ffill().fillna(0)
print("Created final feature set for champion models.")

champion_models = {}
for i, target_col in enumerate(tain_labels_df.columns):
    print(f"  Training final model for {target_col} ({i+1}/{len(tain_labels_df.columns)})")
    y_train_full = tain_labels_df[target_col].dropna()
    X_train_full = features_df_final.loc[y_train_full.index]
    
    if len(X_train_full) < 50:
        champion_models[target_col] = None
        continue

    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, colsample_bytree=0.8)
    model.fit(X_train_full, y_train_full)
    champion_models[target_col] = model
    gc.collect()
print("Successfully trained all 424 final champion models.")

with open('champion_models.pkl', 'wb') as f:
    pickle.dump(champion_models, f)
print("✅ Saved champion_models.pkl to disk.")
print("\n--- TRAINING COMPLETE ---")

