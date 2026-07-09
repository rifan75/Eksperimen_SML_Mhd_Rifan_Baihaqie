"""
Automate Preprocessing - Mhd Rifan Baihaqie
Dataset: Iris (sklearn)
Task: Klasifikasi spesies bunga iris
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_data():
    """Load Iris dataset dan simpan versi raw."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
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


def scale_features(X_train, X_test):
    """Standarisasi fitur menggunakan StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled


def preprocess():
    """Pipeline utama preprocessing yang mengembalikan data siap latih."""
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, '..', 'namadataset_raw')
    output_dir = os.path.join(base_dir, 'namadataset_preprocessing')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    df = load_data()
    df.to_csv(os.path.join(raw_dir, 'iris_raw.csv'), index=False)
    print(f"[INFO] Dataset loaded: {df.shape}")

    # 2. Remove duplicates
    df = remove_duplicates(df)
    print(f"[INFO] After removing duplicates: {df.shape}")

    # 3. Handle outliers
    feature_cols = [c for c in df.columns if c != 'target']
    df = handle_outliers(df, feature_cols)
    print("[INFO] Outliers handled (IQR clipping)")

    # 4. Split features and target
    X = df.drop(columns=['target'])
    y = df['target']

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Standarisasi
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("[INFO] Features scaled (StandardScaler)")

    # 7. Simpan hasil
    train_df = X_train_scaled.copy()
    train_df['target'] = y_train.values
    test_df = X_test_scaled.copy()
    test_df['target'] = y_test.values

    train_df.to_csv(os.path.join(output_dir, 'iris_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'iris_test.csv'), index=False)
    print(f"[INFO] Saved: iris_train.csv ({train_df.shape}), iris_test.csv ({test_df.shape})")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()
    print("[DONE] Preprocessing selesai. Data siap dilatih.")
