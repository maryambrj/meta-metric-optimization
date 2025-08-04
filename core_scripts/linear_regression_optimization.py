#!/usr/bin/env python3
"""
Linear Regression Optimization Script for Meta-Metric Optimization

This script performs meta-metric optimization using linear regression with cross-validation.
Converted from linear_regression.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import os
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import *

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Linear regression optimization")
    parser.add_argument("--dataset", choices=["causal_relations", "hh_rlhf"], 
                       default="causal_relations", help="Dataset to process")
    args = parser.parse_args()
    
    print(f"🚀 Starting linear regression optimization for {args.dataset}...")
    
    # Get dataset-specific configuration
    config = get_dataset_config(args.dataset)
    ranking_dir = get_rankings_dir(args.dataset)
    
    # Configuration
    metrics = ['bleu', 'bleurt', 'meteor', 'rouge']  # Use common metrics for both datasets
    
    # Use dataset-specific configuration
    name_mapping = config['annotator_mapping']
    ordered_columns = config['ordered_columns']
    
    # Load Elo values
    print("📊 Loading Elo rankings...")
    elo = pd.read_csv(os.path.join(ranking_dir, "elo_values.csv"))
    elo = elo.rename(columns=name_mapping)
    elo = elo[ordered_columns]
    
    # === FULL DATASET LINEAR REGRESSION ===
    print("\n🔍 Running full dataset linear regression...")
    
    # Stack metric features
    X = []
    for metric in metrics:
        df = pd.read_csv(os.path.join(ranking_dir, f"{metric}_values.csv"), index_col=0)
        df = df.rename(columns=name_mapping)
        df = df[ordered_columns]
        X.append(df.values.flatten())
    
    X = np.stack(X, axis=1)  # Shape: (200, 4)
    y = elo.values.flatten()  # shape: (samples × annotators,)
    
    # Normalize Elo
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    # Least squares regression with normalized Elo
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y_norm)
    weights = reg.coef_
    
    # Report weights
    print("Optimal weights:")
    for metric, weight in zip(metrics, weights):
        print(f"  {metric}: {weight:.4f}")
    
    # Create combined scores
    combined = np.zeros_like(elo.values)
    for i, metric in enumerate(metrics):
        df = pd.read_csv(os.path.join(ranking_dir, f"{metric}_values.csv"), index_col=0)
        df = df.rename(columns=name_mapping)
        df = df[ordered_columns]
        combined += weights[i] * df.values
    
    # Save final combined score matrix
    combined_df = pd.DataFrame(combined, columns=ordered_columns)
    combined_df.to_csv(os.path.join(ranking_dir, "combined_metric_values.csv"), index=False)
    print("  ✅ Saved combined_metric_values.csv")
    
    # Spearman correlation per row
    spearman_scores = []
    for i in range(elo.shape[0]):
        corr, _ = spearmanr(combined[i], elo.iloc[i])
        spearman_scores.append(corr)
    
    print(f"Average Spearman correlation: {np.mean(spearman_scores):.4f}")
    
    # === LEAVE-SAMPLES-OUT CROSS VALIDATION ===
    print("\n🔄 Running leave-samples-out cross validation...")
    
    # Adapt to dataset size
    num_samples = elo.shape[0]
    test_size = min(2, max(1, num_samples // 10))  # Use 10% or at least 1 sample for testing
    
    all_indices = list(range(num_samples))
    test_indices = random.sample(all_indices, test_size)
    train_indices = [i for i in all_indices if i not in test_indices]
    
    print(f"Test samples: {test_indices}")
    
    # Stack features for all metrics (adapt to dataset size)
    X_all = []
    for metric in metrics:
        df = pd.read_csv(os.path.join(ranking_dir, f"{metric}_values.csv"), index_col=0)
        df = df.rename(columns=name_mapping)
        df = df[ordered_columns]
        X_all.append(df.values)
    
    # Stack: shape (num_metrics, num_samples, num_annotators)
    X_all = np.stack(X_all, axis=0)
    print(f"Feature matrix shape: {X_all.shape}")
    
    # Prepare train and test sets
    num_annotators = len(ordered_columns)
    X_train = X_all[:, train_indices, :].reshape(len(metrics), -1).T  # (train_samples * num_annotators, num_metrics)
    X_test = X_all[:, test_indices, :].reshape(len(metrics), -1).T    # (test_samples * num_annotators, num_metrics)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Prepare target vectors
    y_train = elo.iloc[train_indices].values.flatten()
    y_test = elo.iloc[test_indices].values.flatten()
    
    # Normalize training target
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_norm = (y_train - y_mean) / y_std
    
    # Train linear regression
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y_train_norm)
    
    # Predict on test set
    y_pred = reg.predict(X_test)
    
    # Evaluate Spearman correlations for the test samples
    spearman_scores = []
    for i, idx in enumerate(test_indices):
        pred_row = y_pred[i*num_annotators:(i+1)*num_annotators]
        true_row = y_test[i*num_annotators:(i+1)*num_annotators]
        corr, _ = spearmanr(pred_row, true_row)
        spearman_scores.append(corr)
    
    print(f"Spearman correlations on test samples {test_indices}: {spearman_scores}")
    print(f"Mean test Spearman: {np.mean(spearman_scores):.4f}")
    
    # === BOOTSTRAPPED CROSS VALIDATION ===
    print("\n🔄 Running bootstrapped cross validation...")
    
    n_bootstrap = 100
    bootstrap_scores = []
    
    for bootstrap in range(n_bootstrap):
        # Randomly choose test samples (adaptive to dataset size)
        test_indices = random.sample(all_indices, test_size)
        train_indices = [i for i in all_indices if i not in test_indices]
        
        # Prepare train and test sets
        X_train = X_all[:, train_indices, :].reshape(len(metrics), -1).T
        X_test = X_all[:, test_indices, :].reshape(len(metrics), -1).T
        
        y_train = elo.iloc[train_indices].values.flatten()
        y_test = elo.iloc[test_indices].values.flatten()
        
        # Normalize training target
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        y_train_norm = (y_train - y_mean) / y_std
        
        # Train and predict
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_train, y_train_norm)
        y_pred = reg.predict(X_test)
        
        # Calculate correlations for test samples
        sample_scores = []
        for i, idx in enumerate(test_indices):
            pred_row = y_pred[i*num_annotators:(i+1)*num_annotators]
            true_row = y_test[i*num_annotators:(i+1)*num_annotators]
            corr, _ = spearmanr(pred_row, true_row)
            if not np.isnan(corr):
                sample_scores.append(corr)
        
        if sample_scores:
            bootstrap_scores.append(np.mean(sample_scores))
    
    # Save bootstrapped results
    bootstrap_df = pd.DataFrame({'spearman_correlation': bootstrap_scores})
    bootstrap_df.to_csv(os.path.join(ranking_dir, "spearman_normalized_elo.csv"), index=False)
    print("  ✅ Saved spearman_normalized_elo.csv")
    
    print(f"Bootstrap results:")
    print(f"  Mean: {np.mean(bootstrap_scores):.4f}")
    print(f"  Std: {np.std(bootstrap_scores):.4f}")
    print(f"  95% CI: [{np.percentile(bootstrap_scores, 2.5):.4f}, {np.percentile(bootstrap_scores, 97.5):.4f}]")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(bootstrap_scores), color='red', linestyle='--', label=f'Mean: {np.mean(bootstrap_scores):.4f}')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Spearman Correlations (Bootstrap)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ranking_dir, "bootstrapped_spearman_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved bootstrapped_spearman_plot.png")
    
    print("\n✅ Linear regression optimization complete!")

if __name__ == "__main__":
    main() 