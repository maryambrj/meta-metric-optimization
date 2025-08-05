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
    detailed_scores_file = os.path.join(data_dir, "detailed_scores.csv")
    
    if not os.path.exists(detailed_scores_file):
        print(f"❌ Detailed scores file not found: {detailed_scores_file}")
        print("   This usually means BLEURT checkpoint is missing or metrics calculation failed.")
        print("   Please run the metrics step first: python run_pipeline.py --step metrics --dataset hh_rlhf")
        return
    
    df = pd.read_csv(detailed_scores_file)
    
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
        if len(metric_df) == 0:
            print(f"  ⚠️ No data found for metric: {metric}")
            continue
        pivot = metric_df.pivot(index='Row', columns='Model', values='F1')
        pivot.to_csv(os.path.join(rankings_dir, f"{metric}_values.csv"))
        ranking_tables[metric] = pivot
        print(f"  ✅ Created {metric}_values.csv")
    
    # Process Elo rankings
    elo_file = os.path.join(data_dir, "final_elo_rankings.csv")
    if not os.path.exists(elo_file):
        print(f"❌ Elo rankings file not found: {elo_file}")
        return
        
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
        metric_file = os.path.join(rankings_dir, f"{metric}_values.csv")
        if not os.path.exists(metric_file):
            print(f"  ⚠️ Skipping {metric} - file not found")
            continue
            
        df_metric = pd.read_csv(metric_file, index_col=0)
        df_metric = df_metric.rename(columns=name_mapping)
        
        # Find common columns between elo and metric data
        common_cols = list(set(elo.columns) & set(df_metric.columns))
        if len(common_cols) == 0:
            print(f"  ⚠️ No common columns found for {metric}")
            continue
            
        elo_common = elo[common_cols]
        df_metric_common = df_metric[common_cols]
        
        correlations = []
        
        for i in elo_common.index:
            if i not in df_metric_common.index:
                correlations.append(0.0)
                continue
                
            elo_row = elo_common.loc[i]
            metric_row = df_metric_common.loc[i]
            
            # For HH-RLHF: compare chosen vs rejected directly
            if args.dataset == 'hh_rlhf':
                # Simple correlation between Elo and metric scores
                corr, _ = spearmanr(elo_row, metric_row)
                correlations.append(corr if not pd.isna(corr) else 0.0)
            else:
                # For causal_relations: remove winner and correlate remaining
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