# ppop.py  — First-half training pipeline (clean + reproducible)
import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------ Reproducibility ------------------
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# ------------------ Phase 1: Data Prep ------------------
def load_data(path=r"C:\Users\Faheem\Desktop\Project\Dataset\synthetic_hemophilia_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    df["factor_level_IU_dL"] = df["factor_level_IU_dL"].clip(0, 150)
    df["time_since_last_infusion_hr"] = df["time_since_last_infusion_hr"].clip(0, 168)
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df["tsi_sq"] = df["time_since_last_infusion_hr"] ** 2
    df["log_dose"] = np.log1p(df["infusion_dose_IU"])
    df["dose_per_kg"] = df["infusion_dose_IU"] / df["weight_kg"].replace(0, np.nan)
    df["dose_per_kg"] = df["dose_per_kg"].fillna(df["dose_per_kg"].median())
    # ordinal (not used by models below, but available)
    severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
    df["severity_ord"] = df["severity"].map(severity_map).fillna(1)
    return df

def grouped_split(df, train_size=0.7, val_size=0.15, test_size=0.15,
                  group_col="patient_id", seed=SEED):
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df[group_col]))
    temp = df.iloc[temp_idx].copy()
    val_frac = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_frac, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp, groups=temp[group_col]))
    return (df.iloc[train_idx].reset_index(drop=True),
            temp.iloc[val_idx].reset_index(drop=True),
            temp.iloc[test_idx].reset_index(drop=True))

# ------------------ Phase 2: Models ------------------
def build_preprocessor(num_features, cat_features):
    numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())])
    categorical_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_tf, num_features),
                              ("cat", categorical_tf, cat_features)])

# ---- XGBoost Regression (native Booster + early stopping) ----
def train_xgb_native(train_df, val_df, num_features, cat_features, target="factor_level_IU_dL"):
    preproc = build_preprocessor(num_features, cat_features)
    X_train = train_df[num_features + cat_features]
    y_train = train_df[target].values
    X_val   = val_df[num_features + cat_features]
    y_val   = val_df[target].values

    X_train_proc = preproc.fit_transform(X_train)
    X_val_proc   = preproc.transform(X_val)

    dtrain = xgb.DMatrix(X_train_proc, label=y_train)
    dval   = xgb.DMatrix(X_val_proc,   label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dval, "validation")],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # ✅ Use early stopping info safely
    best_iter = booster.best_iteration if hasattr(booster, "best_iteration") else booster.num_boosted_rounds()
    val_pred = booster.predict(dval, iteration_range=(0, best_iter))

    print(f"[XGB] MAE (val): {mean_absolute_error(y_val, val_pred):.3f} | Best iteration: {best_iter}")

    return preproc, booster   # ✅ ensure return

# ---- Random Forest Classification (bleed risk) ----
def train_rf(train_df, val_df, num_features, cat_features):
    all_num = num_features + ["factor_level_IU_dL"]
    preproc = build_preprocessor(all_num, cat_features)

    X_train = train_df[num_features + cat_features + ["factor_level_IU_dL"]]
    y_train = train_df["bleed_event"].values.astype(int)
    X_val   = val_df[num_features + cat_features + ["factor_level_IU_dL"]]
    y_val   = val_df["bleed_event"].values.astype(int)

    X_train_proc = preproc.fit_transform(X_train)
    X_val_proc   = preproc.transform(X_val)

    clf = RandomForestClassifier(
        n_estimators=600, max_depth=None,
        class_weight="balanced",
        random_state=SEED, n_jobs=-1
    )
    clf.fit(X_train_proc, y_train)
    val_proba = clf.predict_proba(X_val_proc)[:, 1]

    # Threshold sweep to maximize F1 on validation
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.2, 0.8, 37):
        preds = (val_proba >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    auc = roc_auc_score(y_val, val_proba)
    print(f"[RF] AUC (val): {auc:.3f} | Best Thr: {best_thr:.3f} | Best F1: {best_f1:.3f}")
    return preproc, clf, best_thr

# ------------------ Phase 3: LSTM (temporal) ------------------
def _sort_group(grp: pd.DataFrame) -> pd.DataFrame:
    # Keep chronological order if a timestamp-like column exists
    for c in ["timestamp", "record_time", "datetime", "event_time", "time"]:
        if c in grp.columns:
            return grp.sort_values(c)
    return grp  # fallback: as-is

def build_sequences(df, features, target, window=5):
    X, y = [], []
    for _, grp in df.groupby("patient_id"):
        grp = _sort_group(grp)
        arr = grp[features].values
        tgt = grp[target].values
        if len(arr) <= window:
            continue
        for i in range(len(arr) - window):
            X.append(arr[i:i+window])
            y.append(tgt[i+window])
    return np.asarray(X), np.asarray(y)

def train_lstm(train_df, val_df, test_df, seq_features, target="factor_level_IU_dL", window=5):
    scaler = MinMaxScaler()
    # Fit on train only; transform val/test
    train_df[seq_features] = scaler.fit_transform(train_df[seq_features])
    val_df[seq_features]   = scaler.transform(val_df[seq_features])
    test_df[seq_features]  = scaler.transform(test_df[seq_features])

    X_train, y_train = build_sequences(train_df, seq_features, target, window)
    X_val,   y_val   = build_sequences(val_df,   seq_features, target, window)
    X_test,  y_test  = build_sequences(test_df,  seq_features, target, window)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mae")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=200, batch_size=64, callbacks=[es], verbose=0)

    y_pred = model.predict(X_test, verbose=0).reshape(-1, 1)

    # Inverse scaling for interpretability
    def inv_concat(last_x, y_like):
        # last_x: (..., n_features-1), y_like: (..., 1)
        cat = np.concatenate([last_x, y_like], axis=1)
        return scaler.inverse_transform(cat)[:, -1]

    y_pred_rescaled = inv_concat(X_test[:, -1, :-1], y_pred)
    y_test_rescaled = inv_concat(X_test[:, -1, :-1], y_test.reshape(-1, 1))

    mae = np.mean(np.abs(y_pred_rescaled - y_test_rescaled))
    print(f"[LSTM] MAE (test, IU/dL): {mae:.3f}")

    # Save a compact training plot
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_rescaled[:300], label="True", linewidth=1)
    plt.plot(y_pred_rescaled[:300], label="Pred", linewidth=1)
    plt.axhline(50, color="r", linestyle="--", label="Safe 50 IU/dL")
    plt.title("LSTM Temporal Prediction (sample)")
    plt.xlabel("Time steps"); plt.ylabel("Factor level (IU/dL)"); plt.legend()
    plt.tight_layout()
    plt.savefig("lstm_training_plot.png", dpi=160)
    plt.close()

    return model, scaler

# ------------------ Runner ------------------
if __name__ == "__main__":
    df = load_data()
    df = basic_clean(df)
    df = feature_engineer(df)
    train_df, val_df, test_df = grouped_split(df)

    num_features = ["age", "weight_kg", "infusion_dose_IU", "time_since_last_infusion_hr",
                    "tsi_sq", "log_dose", "dose_per_kg"]
    cat_features = ["severity"]

    # Phase 2
    preproc_xgb, xgb_booster = train_xgb_native(train_df, val_df, num_features, cat_features)
    preproc_rf, rf_model, best_thr = train_rf(train_df, val_df, num_features, cat_features)

    # Save Phase 2 artifacts
    xgb_booster.save_model("xgb_booster.json")
    joblib.dump(preproc_xgb, "preproc_xgb.pkl")
    joblib.dump(preproc_rf, "preproc_rf.pkl")
    joblib.dump(rf_model, "rf_model.pkl")
    with open("rf_threshold.txt", "w") as f:
        f.write(str(best_thr))

    # Phase 3
    seq_features = ["infusion_dose_IU", "time_since_last_infusion_hr", "factor_level_IU_dL"]
    lstm_model, lstm_scaler = train_lstm(train_df, val_df, test_df, seq_features)
    lstm_model.save("lstm_model.keras")   # Keras 3 format
    joblib.dump(lstm_scaler, "lstm_scaler.pkl")

    print("Saved artifacts: xgb_booster.json, preproc_xgb.pkl, preproc_rf.pkl, rf_model.pkl, rf_threshold.txt, lstm_model.keras, lstm_scaler.pkl, lstm_training_plot.png")