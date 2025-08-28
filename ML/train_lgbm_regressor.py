#!/usr/bin/env python3
"""
Train a LightGBM regression model on preprocessed fashion-compatibility features.

Inputs
------
training_data.npz
  - X: 2D NumPy array (feature matrix)
  - y: 1D NumPy array (target vector on a 1–5 scale)

Behavior
--------
- Splits data (80/20) with fixed random_state for reproducibility
- Trains a LightGBM regressor (sklearn API)
- Evaluates with Mean Absolute Error (MAE)
- Saves the trained model to 'aesthetic_scorer_v1.pkl'

Console Output (exactly two lines)
----------------------------------
Model Mean Absolute Error (MAE): X.XXX
Model successfully saved to aesthetic_scorer_v1.pkl
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import joblib
import os


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM regressor for fashion compatibility scoring.")
    parser.add_argument(
        "--data",
        default="training_data.npz",
        help="Path to training .npz file containing arrays 'X' and 'y' (default: training_data.npz)",
    )
    parser.add_argument(
        "--out",
        default="aesthetic_scorer_v1.pkl",
        help="Output path for the saved model (default: aesthetic_scorer_v1.pkl)",
    )
    args = parser.parse_args()

    # Load data
    data = np.load(args.data, allow_pickle=False)
    X = data["X"]
    y = data["y"]

    # Train/test split (fixed random state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Define model (quiet by default)
    model = LGBMRegressor(
        random_state=42,
        n_jobs=-1,
        verbose=-1  # keep console clean
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Print ONLY the final MAE line
    print(f"Model Mean Absolute Error (MAE): {mae:.3f}")

    # Save model
    joblib.dump(model, args.out)

    # Print ONLY the save confirmation line
    print(f"Model successfully saved to {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()