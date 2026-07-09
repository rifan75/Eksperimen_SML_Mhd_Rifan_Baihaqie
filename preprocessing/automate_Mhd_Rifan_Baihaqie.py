"""
Automate Preprocessing - Mhd Rifan Baihaqie
Dataset: Heart Disease (UCI Cleveland)
Task: Klasifikasi - memprediksi penyakit jantung (binary)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data():
    """Load Heart Disease dataset dari UCI."""
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    df = pd.read_csv(url, header=None, names=columns, na_values='?')
    # Convert target: 0 = no disease, 1 = disease (binary)
    df['target'] = (df['target'] > 0).astype(int)
    return df


def handle_missing(df):
    """Isi missing values dengan median."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    """Hapus data duplikat."""
    return df.drop_duplicates().reset_index(drop=True)


def handle_outliers(df, feature_cols):
    """Clip outlier menggunakan metode IQR."""
    df = df.copy()
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower, upper)
    return df


def scale_features(X_train, X_test, numeric_cols):
    """Standarisasi fitur numerik menggunakan StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled


def preprocess():
    """Pipeline utama preprocessing yang mengembalikan data siap latih."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, '..', 'namadataset_raw')
    output_dir = os.path.join(base_dir, 'namadataset_preprocessing')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    df = load_data()
    df.to_csv(os.path.join(raw_dir, 'heart_disease_raw.csv'), index=False)
    print(f"[INFO] Dataset loaded: {df.shape}")

    # 2. Handle missing values
    df = handle_missing(df)
    print(f"[INFO] Missing values handled (median imputation)")

    # 3. Remove duplicates
    before = len(df)
    df = remove_duplicates(df)
    print(f"[INFO] Duplicates removed: {before - len(df)} rows")

    # 4. Handle outliers (hanya fitur numerik kontinu)
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df = handle_outliers(df, numeric_cols)
    print("[INFO] Outliers handled (IQR clipping)")

    # 5. Split features and target
    X = df.drop(columns=['target'])
    y = df['target']

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Standarisasi fitur numerik
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, numeric_cols)
    print("[INFO] Features scaled (StandardScaler)")

    # 8. Simpan hasil
    train_df = X_train_scaled.copy().reset_index(drop=True)
    train_df['target'] = y_train.values
    test_df = X_test_scaled.copy().reset_index(drop=True)
    test_df['target'] = y_test.values

    train_df.to_csv(os.path.join(output_dir, 'heart_disease_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'heart_disease_test.csv'), index=False)
    print(f"[INFO] Saved: train ({train_df.shape}), test ({test_df.shape})")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()
    print("[DONE] Preprocessing selesai. Data siap dilatih.")
