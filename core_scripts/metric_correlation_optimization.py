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
    
    # Load Elo rankings (for causal_relations only)
    elo_file = os.path.join(rankings_dir, "elo_values.csv")
    
    if not os.path.exists(elo_file):
        print(f"❌ Elo rankings file not found: {elo_file}")
        return None, None
    
    elo_df = pd.read_csv(elo_file)
    print(f"📊 Loaded Elo file: {elo_file}")
    print(f"📊 Elo DataFrame shape: {elo_df.shape}")
    print(f"📊 Elo DataFrame columns: {list(elo_df.columns)}")
    
    # Load metric scores
    metrics = config['metrics']
    metric_dfs = {}
    
    for metric in metrics:
        metric_file = os.path.join(rankings_dir, f"{metric}_values.csv")
        if os.path.exists(metric_file):
            metric_dfs[metric] = pd.read_csv(metric_file, index_col=0)
            print(f"📊 Loaded {metric} file: {metric_file}")
        else:
            print(f"⚠️ Metric file not found: {metric_file}")
    
    return elo_df, metric_dfs

def prepare_data_for_optimization(elo_df, metric_dfs, dataset_name):
    """Prepare data for linear combination optimization (causal relations only)"""
    print(f"🔧 Preparing data for {dataset_name} dataset...")
    
    # For causal_relations: Elo-based correlation optimization
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

def save_results(weights, metrics, correlations, dataset_name, data):
    """Save comprehensive optimization results"""
    config = get_dataset_config(dataset_name)
    rankings_dir = get_rankings_dir(dataset_name)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Metric': metrics,
        'Optimal_Weight': weights,
        'Mean_CV_Correlation': np.mean(correlations),
        'Std_CV_Correlation': np.std(correlations)
    })
    
    # Save basic results
    results_file = os.path.join(rankings_dir, "linear_optimization_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"💾 Saved optimization results to {results_file}")
    
    # Create combined metric values with optimal weights
    print("📊 Creating combined metric values...")
    combined_scores = np.zeros(len(data))
    for i, metric in enumerate(metrics):
        if metric in data.columns:
            combined_scores += weights[i] * data[metric].values
    
    # Create combined metric DataFrame
    combined_df = pd.DataFrame({
        'sample_id': data.index,
        'combined_score': combined_scores,
        'elo_score': data['elo'].values
    })
    
    # Add individual metric scores
    for metric in metrics:
        if metric in data.columns:
            combined_df[f'{metric}_score'] = data[metric].values
    
    # Save combined metric values
    combined_file = os.path.join(rankings_dir, "combined_metric_values.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"💾 Saved combined metric values to {combined_file}")
    
    # Create normalized Spearman correlations
    print("📈 Creating normalized Spearman correlations...")
    spearman_data = []
    for i, metric in enumerate(metrics):
        if metric in data.columns:
            corr, _ = spearmanr(data[metric].values, data['elo'].values)
            spearman_data.append({
                'Metric': metric,
                'Spearman_Correlation': corr if not np.isnan(corr) else 0.0,
                'Optimal_Weight': weights[i]
            })
    
    spearman_df = pd.DataFrame(spearman_data)
    spearman_file = os.path.join(rankings_dir, "spearman_normalized_elo.csv")
    spearman_df.to_csv(spearman_file, index=False)
    print(f"💾 Saved Spearman correlations to {spearman_file}")
    
    # Create visualization plot
    print("📊 Creating visualization plot...")
    create_visualization_plot(spearman_df, combined_df, rankings_dir, dataset_name)
    
    # Print summary
    print(f"\n📋 Metric Correlation Optimization Summary:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Mean CV Correlation: {np.mean(correlations):.4f}")
    print(f"   Optimal Weights:")
    for metric, weight in zip(metrics, weights):
        print(f"     {metric}: {weight:.4f}")
    
    return results_df

def create_visualization_plot(spearman_df, combined_df, rankings_dir, dataset_name):
    """Create comprehensive visualization plot"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Metric Correlation Optimization Results: {dataset_name.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: Spearman correlations by metric
    metrics = spearman_df['Metric']
    correlations = spearman_df['Spearman_Correlation']
    weights = spearman_df['Optimal_Weight']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, correlations, width, label='Spearman Correlation', alpha=0.8)
    bars2 = ax1.bar(x + width/2, weights, width, label='Optimal Weight', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Correlation / Weight')
    ax1.set_title('Spearman Correlation vs Optimal Weights')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Combined score vs Elo correlation
    ax2.scatter(combined_df['combined_score'], combined_df['elo_score'], alpha=0.6, s=50)
    ax2.set_xlabel('Combined Metric Score')
    ax2.set_ylabel('Elo Score')
    ax2.set_title('Combined Metric Score vs Elo Score')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(combined_df['combined_score'], combined_df['elo_score'], 1)
    p = np.poly1d(z)
    ax2.plot(combined_df['combined_score'], p(combined_df['combined_score']), "r--", alpha=0.8)
    
    # Plot 3: Individual metric correlations
    metric_cols = [col for col in combined_df.columns if col.endswith('_score') and col != 'combined_score']
    metric_names = [col.replace('_score', '') for col in metric_cols]
    
    correlations_individual = []
    for col in metric_cols:
        corr, _ = spearmanr(combined_df[col], combined_df['elo_score'])
        correlations_individual.append(corr if not np.isnan(corr) else 0.0)
    
    bars = ax3.bar(metric_names, correlations_individual, alpha=0.8, color='skyblue')
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Spearman Correlation')
    ax3.set_title('Individual Metric Correlations with Elo')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Weight distribution
    if weights.sum() > 0:  # Only create pie chart if there are non-zero weights
        ax4.pie(weights, labels=metrics, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Optimal Weight Distribution')
    else:
        ax4.text(0.5, 0.5, 'All weights are zero', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Weight Distribution')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(rankings_dir, "bootstrapped_spearman_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization plot to {plot_file}")
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Metric correlation optimization")
    parser.add_argument("--dataset", choices=["causal_relations"], 
                       default="causal_relations", help="Dataset to process")
    args = parser.parse_args()
    
    # Use correlation-based optimization for causal relations
    print(f"🚀 Correlation-based Optimization for {args.dataset}")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    elo_df, metric_dfs = load_data(args.dataset)
    
    if elo_df is None or not metric_dfs:
        print("❌ Failed to load data")
        return
    
    # Prepare data for optimization
    print("🔧 Preparing data for optimization...")
    data = prepare_data_for_optimization(elo_df, metric_dfs, args.dataset)
    
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
    save_results(optimal_weights, metrics, correlations, args.dataset, data)
    
    print(f"\n✅ Correlation-based optimization complete!")
    print(f"   Final correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"   Optimal weights: {dict(zip(metrics, optimal_weights))}")

if __name__ == "__main__":
    main()