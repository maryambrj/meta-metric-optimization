#!/usr/bin/env python3
"""
Pairwise Logistic Regression for Summary Data

This module implements pairwise logistic regression for datasets with binary preferences
(like summarize-from-feedback) where we have winner/loser pairs rather than full rankings.

Instead of optimizing rank correlation like with causal data, we train a logistic regression
that predicts the winner from the difference of metric vectors.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import spearmanr
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import get_dataset_config, get_data_dir, get_rankings_dir

def prepare_pairwise_data(elo_df, metric_dfs, dataset_name):
    """
    Prepare pairwise data for logistic regression
    
    Args:
        elo_df: DataFrame with winner/loser Elo scores
        metric_dfs: Dictionary of metric DataFrames
        dataset_name: Name of dataset
    
    Returns:
        tuple: (X_delta, y, sample_info) where:
            - X_delta: Difference vectors (winner_metrics - loser_metrics)
            - y: Binary labels (1 for correct preference prediction)
            - sample_info: Sample metadata
    """
    print(f"🔧 Preparing pairwise data for {dataset_name}...")
    
    # Initialize lists to store pairwise data
    X_delta = []
    y = []
    sample_info = []
    
    # Get available metrics
    metrics = list(metric_dfs.keys())
    print(f"📊 Available metrics: {metrics}")
    
    # Process each sample (winner/loser pair)
    for idx, row in elo_df.iterrows():
        sample_data = {'sample_id': row['sample_id']}
        
        # Get metric vectors for winner and loser
        winner_metrics = []
        loser_metrics = []
        valid_metrics = []
        
        for metric in metrics:
            metric_df = metric_dfs[metric]
            if metric_df is not None and 'winner' in metric_df.columns and 'loser' in metric_df.columns:
                # Find the corresponding metric values for this sample
                metric_row = metric_df[metric_df.index == idx]
                if len(metric_row) > 0:
                    winner_val = metric_row['winner'].iloc[0]
                    loser_val = metric_row['loser'].iloc[0]
                    
                    if not (np.isnan(winner_val) or np.isnan(loser_val)):
                        winner_metrics.append(winner_val)
                        loser_metrics.append(loser_val)
                        valid_metrics.append(metric)
        
        # Skip if we don't have enough valid metrics
        if len(valid_metrics) < 2:
            continue
        
        # Create difference vectors for both directions
        winner_metrics = np.array(winner_metrics)
        loser_metrics = np.array(loser_metrics)
        
        # Direction 1: winner - loser (label = 1, winner preferred)
        delta_1 = winner_metrics - loser_metrics
        X_delta.append(delta_1)
        y.append(1)
        
        sample_data_1 = sample_data.copy()
        sample_data_1['winner_elo'] = row.get('winner', 0)
        sample_data_1['loser_elo'] = row.get('loser', 0) 
        sample_data_1['metrics'] = valid_metrics
        sample_data_1['direction'] = 'winner_preferred'
        sample_info.append(sample_data_1)
        
        # Direction 2: loser - winner (label = 0, loser not preferred)
        delta_2 = loser_metrics - winner_metrics
        X_delta.append(delta_2)
        y.append(0)
        
        sample_data_2 = sample_data.copy()
        sample_data_2['winner_elo'] = row.get('loser', 0)  # Swapped
        sample_data_2['loser_elo'] = row.get('winner', 0)  # Swapped
        sample_data_2['metrics'] = valid_metrics
        sample_data_2['direction'] = 'loser_not_preferred'
        sample_info.append(sample_data_2)
    
    X_delta = np.array(X_delta)
    y = np.array(y)
    
    print(f"✅ Prepared {len(X_delta)} pairwise samples")
    print(f"📊 Feature dimensions: {X_delta.shape[1] if len(X_delta) > 0 else 0}")
    print(f"📊 Valid metrics: {valid_metrics}")
    
    return X_delta, y, sample_info, valid_metrics

def train_pairwise_logistic_regression(X_delta, y, metrics, regularization='l2', C=1.0):
    """
    Train logistic regression on pairwise preference data
    
    Args:
        X_delta: Difference vectors (winner - loser)
        y: Binary labels (should be all 1s for winner preference)
        metrics: List of metric names
        regularization: Regularization type ('l1', 'l2', or 'none')
        C: Regularization strength (lower = more regularization)
    
    Returns:
        tuple: (model, weights, training_accuracy)
    """
    print("🔧 Training pairwise logistic regression...")
    
    if len(X_delta) == 0:
        print("❌ No training data available")
        return None, None, 0.0
    
    # Configure regularization
    if regularization == 'none':
        penalty = None
        solver = 'lbfgs'
    else:
        penalty = regularization
        solver = 'liblinear' if regularization == 'l1' else 'lbfgs'
    
    # Train logistic regression
    if penalty is None:
        model = LogisticRegression(
            solver=solver, 
            max_iter=1000, 
            random_state=42,
            fit_intercept=False  # No intercept needed for difference vectors
        )
    else:
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver, 
            max_iter=1000, 
            random_state=42,
            fit_intercept=False  # No intercept needed for difference vectors
        )
    
    model.fit(X_delta, y)
    
    # Get weights (coefficients)
    weights = model.coef_[0]
    
    # Calculate training accuracy
    y_pred = model.predict(X_delta)
    training_accuracy = accuracy_score(y, y_pred)
    
    print(f"✅ Training complete")
    print(f"   Training accuracy: {training_accuracy:.4f}")
    print(f"   Weights: {dict(zip(metrics, weights))}")
    
    return model, weights, training_accuracy

def cross_validate_pairwise_model(X_delta, y, metrics, n_splits=5, regularization='l2', C=1.0):
    """
    Cross-validate the pairwise logistic regression model
    
    Args:
        X_delta: Difference vectors
        y: Binary labels  
        metrics: List of metric names
        n_splits: Number of CV folds
        regularization: Regularization type
        C: Regularization strength
    
    Returns:
        tuple: (mean_weights, cv_scores, all_weights)
    """
    print(f"🔄 Cross-validating pairwise model with {n_splits}-fold CV...")
    
    if len(X_delta) == 0:
        return np.zeros(len(metrics)), [], []
    
    # Use Leave-One-Out for small datasets
    if len(X_delta) <= 10:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
        n_splits = len(X_delta)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=min(n_splits, len(X_delta)), shuffle=True, random_state=42)
    
    cv_scores = []
    all_weights = []
    
    for train_idx, test_idx in cv.split(X_delta):
        X_train, X_test = X_delta[train_idx], X_delta[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model on fold
        model, weights, _ = train_pairwise_logistic_regression(
            X_train, y_train, metrics, regularization, C
        )
        
        if model is not None:
            # Evaluate on test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores.append(accuracy)
            all_weights.append(weights)
        else:
            cv_scores.append(0.0)
            all_weights.append(np.zeros(len(metrics)))
    
    # Calculate mean weights and scores
    mean_weights = np.mean(all_weights, axis=0) if all_weights else np.zeros(len(metrics))
    
    print(f"📊 Cross-validation results:")
    print(f"   Mean accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"   Min accuracy: {np.min(cv_scores):.4f}")
    print(f"   Max accuracy: {np.max(cv_scores):.4f}")
    
    return mean_weights, cv_scores, all_weights

def evaluate_pairwise_preference_accuracy(weights, metric_dfs, elo_df, metrics):
    """
    Evaluate how well the weighted combination predicts pairwise preferences
    
    Args:
        weights: Optimized weights for metrics
        metric_dfs: Dictionary of metric DataFrames
        elo_df: Elo DataFrame with winner/loser pairs
        metrics: List of metric names
    
    Returns:
        tuple: (accuracy, predictions_df)
    """
    print("📊 Evaluating pairwise preference accuracy...")
    
    correct_predictions = 0
    total_predictions = 0
    predictions_data = []
    
    for idx, row in elo_df.iterrows():
        sample_id = row['sample_id']
        
        # Get metric values for winner and loser
        winner_score = 0
        loser_score = 0
        valid_metrics = []
        
        for i, metric in enumerate(metrics):
            metric_df = metric_dfs[metric]
            if metric_df is not None and 'winner' in metric_df.columns and 'loser' in metric_df.columns:
                metric_row = metric_df[metric_df.index == idx]
                if len(metric_row) > 0:
                    winner_val = metric_row['winner'].iloc[0]
                    loser_val = metric_row['loser'].iloc[0]
                    
                    if not (np.isnan(winner_val) or np.isnan(loser_val)):
                        winner_score += weights[i] * winner_val
                        loser_score += weights[i] * loser_val
                        valid_metrics.append(metric)
        
        # Skip if no valid metrics
        if len(valid_metrics) == 0:
            continue
        
        # Predict preference: winner should have higher combined score
        predicted_winner = winner_score > loser_score
        actual_winner = True  # Winner is always preferred in our data
        
        correct = predicted_winner == actual_winner
        correct_predictions += int(correct)
        total_predictions += 1
        
        predictions_data.append({
            'sample_id': sample_id,
            'winner_combined_score': winner_score,
            'loser_combined_score': loser_score,
            'predicted_winner': predicted_winner,
            'actual_winner': actual_winner,
            'correct': correct,
            'score_difference': winner_score - loser_score
        })
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    predictions_df = pd.DataFrame(predictions_data)
    
    print(f"✅ Preference prediction results:")
    print(f"   Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"   Mean score difference: {predictions_df['score_difference'].mean():.4f}")
    
    return accuracy, predictions_df

def save_pairwise_results(weights, metrics, cv_scores, predictions_df, dataset_name):
    """
    Save pairwise logistic regression results
    
    Args:
        weights: Optimized weights
        metrics: List of metric names  
        cv_scores: Cross-validation scores
        predictions_df: Predictions DataFrame
        dataset_name: Name of dataset
    """
    rankings_dir = get_rankings_dir(dataset_name)
    
    # Save weights and CV results
    results_df = pd.DataFrame({
        'Metric': metrics,
        'Optimal_Weight': weights,
        'Mean_CV_Accuracy': np.mean(cv_scores),
        'Std_CV_Accuracy': np.std(cv_scores)
    })
    
    results_file = os.path.join(rankings_dir, "pairwise_logistic_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"💾 Saved pairwise logistic results to {results_file}")
    
    # Save predictions
    predictions_file = os.path.join(rankings_dir, "pairwise_predictions.csv") 
    predictions_df.to_csv(predictions_file, index=False)
    print(f"💾 Saved predictions to {predictions_file}")
    
    # Create combined metric values (compatible with existing analysis)
    combined_df = pd.DataFrame({
        'sample_id': predictions_df['sample_id'],
        'combined_score_winner': predictions_df['winner_combined_score'],
        'combined_score_loser': predictions_df['loser_combined_score'],
        'score_difference': predictions_df['score_difference'],
        'predicted_correctly': predictions_df['correct']
    })
    
    combined_file = os.path.join(rankings_dir, "combined_metric_values.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"💾 Saved combined metric values to {combined_file}")
    
    # Create visualization
    create_pairwise_visualization(results_df, predictions_df, rankings_dir, dataset_name)
    
    return results_df

def create_pairwise_visualization(results_df, predictions_df, rankings_dir, dataset_name):
    """
    Create visualization plots for pairwise logistic regression results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Pairwise Logistic Regression Results: {dataset_name.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: Optimal weights
    metrics = results_df['Metric']
    weights = results_df['Optimal_Weight']
    
    bars = ax1.bar(metrics, weights, alpha=0.8, color='skyblue')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Optimal Weight')
    ax1.set_title('Optimal Weights from Pairwise Logistic Regression')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Score differences distribution
    ax2.hist(predictions_df['score_difference'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='No preference')
    ax2.set_xlabel('Winner Score - Loser Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Combined Score Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy by score difference
    score_diff_bins = pd.cut(predictions_df['score_difference'], bins=10)
    accuracy_by_bin = predictions_df.groupby(score_diff_bins)['correct'].mean()
    bin_centers = [interval.mid for interval in accuracy_by_bin.index]
    
    ax3.plot(bin_centers, accuracy_by_bin.values, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Score Difference (binned)')
    ax3.set_ylabel('Prediction Accuracy')
    ax3.set_title('Prediction Accuracy vs Score Difference')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # Plot 4: Weight distribution pie chart (using absolute values)
    abs_weights = np.abs(weights)
    if abs_weights.sum() > 0:  # Only create pie chart if there are non-zero weights
        ax4.pie(abs_weights, labels=metrics, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Weight Distribution (Absolute Values)')
    else:
        ax4.text(0.5, 0.5, 'All weights are zero', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Weight Distribution')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(rankings_dir, "pairwise_logistic_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization to {plot_file}")
    plt.close()

def main():
    """Main function for testing pairwise logistic regression"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pairwise Logistic Regression")
    parser.add_argument("--dataset", choices=["summarize_feedback", "hh_rlhf"], 
                       default="summarize_feedback", help="Dataset to process")
    parser.add_argument("--regularization", choices=["l1", "l2", "none"], 
                       default="l2", help="Regularization type")
    parser.add_argument("--C", type=float, default=1.0, 
                       help="Regularization strength")
    
    args = parser.parse_args()
    
    print(f"🚀 Pairwise Logistic Regression for {args.dataset}")
    print("=" * 60)
    
    # Load data (reuse logic from linear_optimization.py)
    from linear_optimization import load_data
    elo_df, metric_dfs = load_data(args.dataset)
    
    if elo_df is None or not metric_dfs:
        print("❌ Failed to load data")
        return
    
    # Prepare pairwise data
    X_delta, y, sample_info, metrics = prepare_pairwise_data(elo_df, metric_dfs, args.dataset)
    
    if len(X_delta) == 0:
        print("❌ No pairwise data available")
        return
    
    # Cross-validate model
    weights, cv_scores, _ = cross_validate_pairwise_model(
        X_delta, y, metrics, regularization=args.regularization, C=args.C
    )
    
    # Evaluate preference accuracy
    accuracy, predictions_df = evaluate_pairwise_preference_accuracy(
        weights, metric_dfs, elo_df, metrics
    )
    
    # Save results
    results_df = save_pairwise_results(weights, metrics, cv_scores, predictions_df, args.dataset)
    
    print(f"\n✅ Pairwise logistic regression complete!")
    print(f"   Final accuracy: {accuracy:.4f}")
    print(f"   Optimal weights: {dict(zip(metrics, weights))}")

if __name__ == "__main__":
    main()