"""
=============================================================================
SMART BUILDING — STEP 1: TRAIN MODEL
=============================================================================
Run this once, or whenever you have new/updated data.
Saves the trained XGBoost model to disk so run_optimization.py
can load it without retraining.

Usage:
    python train_model.py
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- Paths are relative to this file, so the repo works from any directory ---
ROOT  = Path(__file__).resolve().parent
BASE  = ROOT / "data" / "cleaned"
OUT   = ROOT / "outputs"
MODEL = ROOT / "models" / "model.pkl"
OUT.mkdir(parents=True, exist_ok=True)
MODEL.parent.mkdir(parents=True, exist_ok=True)
# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "="*60)
print("STEP 1 — Loading prepared datasets")
print("="*60)

train  = pd.read_csv(f"{BASE}/train_scaled.csv", parse_dates=['timestamp'])
val    = pd.read_csv(f"{BASE}/val_scaled.csv",   parse_dates=['timestamp'])
test   = pd.read_csv(f"{BASE}/test_scaled.csv",  parse_dates=['timestamp'])
scaler = pd.read_csv(f"{BASE}/scaler_params.csv", index_col=0)

TARGET   = 'total_load_kW'
DROP     = ['timestamp', TARGET]
FEATURES = [c for c in train.columns if c not in DROP]

X_train = train[FEATURES].values
y_train = train[TARGET].values
X_val   = val[FEATURES].values
y_val   = val[TARGET].values
X_test  = test[FEATURES].values
y_test  = test[TARGET].values

print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"  Features ({len(FEATURES)}): {FEATURES[:8]}...")

# Inverse scale helper
tgt_min = scaler.loc[TARGET, 'min']
tgt_rng = scaler.loc[TARGET, 'range']

def inverse_scale(arr):
    return arr * tgt_rng + tgt_min

# =============================================================================
# TRAIN XGBOOST
# =============================================================================
print("\n" + "="*60)
print("STEP 2 — Training XGBoost Load Prediction Model")
print("="*60)

xgb_model = xgb.XGBRegressor(
    n_estimators          = 800,
    max_depth             = 7,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 3,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    random_state          = 42,
    n_jobs                = -1,
    early_stopping_rounds = 30,
    eval_metric           = 'rmse'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# =============================================================================
# EVALUATE
# =============================================================================
def metrics(true, pred, label):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae  = mean_absolute_error(true, pred)
    r2   = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-6))) * 100
    print(f"  [{label}] RMSE={rmse:.4f} kW | MAE={mae:.4f} kW | R²={r2:.4f} | MAPE={mape:.2f}%")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

y_pred_val  = xgb_model.predict(X_val)
y_pred_test = xgb_model.predict(X_test)

metrics(inverse_scale(y_val),  inverse_scale(y_pred_val),  "Validation")
metrics(inverse_scale(y_test), inverse_scale(y_pred_test), "Test      ")

# Feature importance
feat_imp = pd.DataFrame({
    'feature':    FEATURES,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 10 features:")
for _, row in feat_imp.head(10).iterrows():
    bar = "█" * int(row['importance'] * 300)
    print(f"    {row['feature']:<30} {row['importance']:.4f}  {bar}")

# =============================================================================
# SAVE MODEL + METADATA
# =============================================================================
print("\n" + "="*60)
print("STEP 3 — Saving model to disk")
print("="*60)

payload = {
    'model':    xgb_model,
    'features': FEATURES,
    'tgt_min':  tgt_min,
    'tgt_rng':  tgt_rng,
}

joblib.dump(payload, MODEL)
feat_imp.to_csv(OUT / "feature_importance.csv", index=False)
print(f"  Model saved → {MODEL}")
print(f"  Feature importances → {OUT / 'feature_importance.csv'}")
print(f"  Size: {os.path.getsize(MODEL) / 1024:.1f} KB")
print("\n  Done! Now run run_optimization.py to optimize schedules.")
