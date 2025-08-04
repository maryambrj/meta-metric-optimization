#!/usr/bin/env python3
"""
Regression Test Script for Meta-Metric Optimization

This script processes detailed scores and creates ranking tables with Spearman correlations.
Converted from reg_test.ipynb
"""

import pandas as pd
import os
from scipy.stats import spearmanr
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import *

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regression test processing")
    parser.add_argument("--dataset", choices=["causal_relations", "hh_rlhf"], 
                       default="causal_relations", help="Dataset to process")
    args = parser.parse_args()
    
    print(f"🔄 Starting regression test processing for {args.dataset}...")
    
    # Get dataset-specific paths
    config = get_dataset_config(args.dataset)
    data_dir = get_data_dir(args.dataset)
    rankings_dir = get_rankings_dir(args.dataset)
    
    # Load detailed scores
    df = pd.read_csv(os.path.join(data_dir, "detailed_scores.csv"))
    
    # Clean up unnamed columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # Define metrics and create ranking tables
    ranking_tables = {}
    metrics = config['metrics']
    
    # Create output directory
    os.makedirs(rankings_dir, exist_ok=True)
    
    print("📊 Creating ranking tables for each metric...")
    for metric in metrics:
        metric_df = df[df['Metric'] == metric]
        pivot = metric_df.pivot(index='Row', columns='Model', values='F1')
        pivot.to_csv(os.path.join(rankings_dir, f"{metric}_values.csv"))
        ranking_tables[metric] = pivot
        print(f"  ✅ Created {metric}_values.csv")
    
    # Process Elo rankings
    elo_file = os.path.join(data_dir, "final_elo_rankings.csv")
    elo_df = pd.read_csv(elo_file)
    elo_df = elo_df.drop(columns=["winner", "sentence_start"], errors="ignore")
    elo_df.to_csv(os.path.join(rankings_dir, "elo_values.csv"), index=False)
    print("  ✅ Created elo_values.csv")
    
    # Use dataset-specific name mapping
    name_mapping = config['annotator_mapping']
    
    elo = pd.read_csv(os.path.join(rankings_dir, "elo_values.csv"))
    elo = elo.rename(columns=name_mapping)
    
    # Compute Spearman correlation per metric, removing the top-scoring model (winner) for each row
    print("📈 Computing Spearman correlations...")
    spearman_scores = {}
    
    for metric in metrics:
        df_metric = pd.read_csv(os.path.join(rankings_dir, f"{metric}_values.csv"), index_col=0)
        df_metric = df_metric.rename(columns=name_mapping)
        df_metric = df_metric[elo.columns]  # ensure consistent column order
        
        correlations = []
        
        for i in elo.index:
            elo_row = elo.loc[i]
            metric_row = df_metric.loc[i]
            
            winner = elo_row.idxmax()
            elo_filtered = elo_row.drop(winner)
            metric_filtered = metric_row.drop(winner)
            
            corr, _ = spearmanr(elo_filtered, metric_filtered)
            correlations.append(corr if not pd.isna(corr) else 0.0)
        
        spearman_scores[metric] = correlations
        print(f"  ✅ Computed correlations for {metric}")
    
    # Save Spearman correlations
    spearman_df = pd.DataFrame(spearman_scores)
    spearman_df.to_csv(os.path.join(rankings_dir, "spearman_per_sample.csv"), index=False)
    print("  ✅ Saved spearman_per_sample.csv")
    
    # Print summary statistics
    print("\n📊 Spearman Correlation Summary:")
    for metric in metrics:
        mean_corr = pd.Series(spearman_scores[metric]).mean()
        print(f"  {metric}: {mean_corr:.4f}")
    
    print("\n✅ Regression test processing complete!")

if __name__ == "__main__":
    main() 