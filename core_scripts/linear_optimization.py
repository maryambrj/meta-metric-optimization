import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import os
import sys
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import get_dataset_config, get_data_dir, get_rankings_dir

def load_data(dataset_name):
    """Load Elo rankings and metric scores"""
    config = get_dataset_config(dataset_name)
    data_dir = get_data_dir(dataset_name)
    rankings_dir = get_rankings_dir(dataset_name)
    
    # Load Elo rankings
    elo_file = os.path.join(rankings_dir, "elo_values.csv")
    if not os.path.exists(elo_file):
        print(f"❌ Elo rankings file not found: {elo_file}")
        return None, None
    
    elo_df = pd.read_csv(elo_file)
    
    # Load metric scores
    metrics = config['metrics']
    metric_dfs = {}
    
    for metric in metrics:
        metric_file = os.path.join(rankings_dir, f"{metric}_values.csv")
        if os.path.exists(metric_file):
            metric_dfs[metric] = pd.read_csv(metric_file, index_col=0)
        else:
            print(f"⚠️ Metric file not found: {metric_file}")
    
    return elo_df, metric_dfs

def prepare_data_for_optimization(elo_df, metric_dfs):
    """Prepare data for linear combination optimization"""
    # Get common samples between Elo and metrics
    elo_samples = elo_df.index.tolist()
    
    # For each metric, get scores for all samples
    metric_scores = {}
    for metric, df in metric_dfs.items():
        if df is not None:
            # Get scores for all models/annotators
            common_cols = list(set(elo_df.columns) & set(df.columns))
            if common_cols:
                # Average scores across models for each sample
                metric_scores[metric] = df[common_cols].mean(axis=1)
    
    # Create combined DataFrame
    combined_data = pd.DataFrame(index=elo_samples)
    
    # Add Elo rankings (average across models)
    combined_data['elo'] = elo_df.mean(axis=1)
    
    # Add metric scores
    for metric, scores in metric_scores.items():
        combined_data[metric] = scores
    
    return combined_data

def objective_function(weights, data, metrics):
    """Objective function: negative Spearman correlation"""
    if len(weights) != len(metrics):
        return 1.0  # Return worst correlation if dimensions don't match
    
    # Calculate linear combination
    combined_score = np.zeros(len(data))
    for i, metric in enumerate(metrics):
        if metric in data.columns:
            combined_score += weights[i] * data[metric].values
    
    # Calculate Spearman correlation with Elo
    correlation, _ = spearmanr(combined_score, data['elo'].values)
    
    # Return negative correlation (minimize negative = maximize positive)
    return -correlation if not np.isnan(correlation) else 1.0

def optimize_weights(data, metrics, method='SLSQP'):
    """Optimize weights for linear combination"""
    print(f"🔧 Optimizing weights for metrics: {metrics}")
    
    # Initial weights (equal weights)
    initial_weights = np.ones(len(metrics)) / len(metrics)
    
    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in metrics]
    
    # Optimize
    result = minimize(
        objective_function,
        initial_weights,
        args=(data, metrics),
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if result.success:
        print(f"✅ Optimization successful!")
        print(f"   Final correlation: {-result.fun:.4f}")
        return result.x
    else:
        print(f"⚠️ Optimization failed: {result.message}")
        return initial_weights

def cross_validate_weights(data, metrics, n_splits=5):
    """Cross-validate the optimized weights"""
    print(f"🔄 Cross-validating weights with {n_splits}-fold CV...")
    
    correlations = []
    weights_list = []
    
    # Use Leave-One-Out for small datasets
    if len(data) <= 10:
        cv = LeaveOneOut()
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=min(n_splits, len(data)), shuffle=True, random_state=42)
    
    for train_idx, test_idx in cv.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Optimize weights on training data
        weights = optimize_weights(train_data, metrics)
        weights_list.append(weights)
        
        # Test on validation data
        combined_score = np.zeros(len(test_data))
        for i, metric in enumerate(metrics):
            if metric in test_data.columns:
                combined_score += weights[i] * test_data[metric].values
        
        correlation, _ = spearmanr(combined_score, test_data['elo'].values)
        correlations.append(correlation if not np.isnan(correlation) else 0.0)
    
    print(f"📊 Cross-validation results:")
    print(f"   Mean correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"   Min correlation: {np.min(correlations):.4f}")
    print(f"   Max correlation: {np.max(correlations):.4f}")
    
    return np.mean(weights_list, axis=0), correlations

def save_results(weights, metrics, correlations, dataset_name):
    """Save optimization results"""
    config = get_dataset_config(dataset_name)
    rankings_dir = get_rankings_dir(dataset_name)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Metric': metrics,
        'Optimal_Weight': weights,
        'Mean_CV_Correlation': np.mean(correlations),
        'Std_CV_Correlation': np.std(correlations)
    })
    
    # Save results
    results_file = os.path.join(rankings_dir, "linear_optimization_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"💾 Saved optimization results to {results_file}")
    
    # Print summary
    print(f"\n📋 Linear Combination Optimization Summary:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Mean CV Correlation: {np.mean(correlations):.4f}")
    print(f"   Optimal Weights:")
    for metric, weight in zip(metrics, weights):
        print(f"     {metric}: {weight:.4f}")
    
    return results_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Linear combination optimization")
    parser.add_argument("--dataset", choices=["causal_relations", "hh_rlhf"], 
                       default="hh_rlhf", help="Dataset to process")
    args = parser.parse_args()
    
    print(f"🚀 Linear Combination Optimization for {args.dataset}")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    elo_df, metric_dfs = load_data(args.dataset)
    
    if elo_df is None or not metric_dfs:
        print("❌ Failed to load data")
        return
    
    # Prepare data for optimization
    print("🔧 Preparing data for optimization...")
    data = prepare_data_for_optimization(elo_df, metric_dfs)
    
    if data is None or len(data) == 0:
        print("❌ No data available for optimization")
        return
    
    print(f"📈 Data shape: {data.shape}")
    print(f"📊 Available metrics: {list(data.columns[1:])}")  # Exclude 'elo'
    
    # Get metrics (exclude 'elo' column)
    metrics = [col for col in data.columns if col != 'elo']
    
    if len(metrics) == 0:
        print("❌ No metrics available for optimization")
        return
    
    # Cross-validate weights
    optimal_weights, correlations = cross_validate_weights(data, metrics)
    
    # Save results
    results_df = save_results(optimal_weights, metrics, correlations, args.dataset)
    
    print("\n✅ Linear combination optimization complete!")

if __name__ == "__main__":
    main() 