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
- Trains a LightGBM regressor (sklearn API) with early stopping
- Evaluates with Mean Absolute Error (MAE)
- Saves the trained model to the path provided via --out (default: 'aesthetic_scorer_v1.pkl')

Console Output (exactly two lines)
----------------------------------
Model Mean Absolute Error (MAE): X.XXX
Model successfully saved to aesthetic_scorer_v1.pkl
"""

import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import joblib


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
    # Training controls / hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--test-size", type=float, default=0.20, help="Validation split size (default: 0.20)")
    parser.add_argument("--n-estimators", type=int, default=3000, help="Max boosting rounds (default: 3000)")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate (default: 0.03)")
    parser.add_argument("--num-leaves", type=int, default=63, help="num_leaves (default: 63)")
    parser.add_argument("--min-child-samples", type=int, default=20, help="min_child_samples (default: 20)")
    parser.add_argument("--subsample", type=float, default=0.8, help="subsample/bagging_fraction (default: 0.8)")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="feature_fraction (default: 0.8)")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization (default: 0.0)")
    parser.add_argument("--reg-lambda", type=float, default=0.0, help="L2 regularization (default: 0.0)")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="Early stopping rounds (default: 100)")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data, allow_pickle=False)
    X = data["X"]
    y = data["y"]

    # Train/test split (fixed random state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Define model (quiet) — use params that generally work well for tabular + embeddings
    model = LGBMRegressor(
        random_state=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        n_jobs=-1,
        verbose=-1,  # keep console quiet; we will enforce 2-line output
    )

    # Train with early stopping on the validation fold
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        callbacks=[
            early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
            log_evaluation(period=0),  # silence internal logs
        ],
    )

    # Evaluate
    # Use best_iteration_ if available (set by early stopping)
    try:
        best_iter = model.best_iteration_
        preds = model.predict(X_test, num_iteration=best_iter)
    except Exception:
        preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Print ONLY the final MAE line
    print(f"Model Mean Absolute Error (MAE): {mae:.3f}")

    # Save model
    joblib.dump(model, args.out)

    # Print ONLY the save confirmation line (use basename to match example)
    print(f"Model successfully saved to {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()