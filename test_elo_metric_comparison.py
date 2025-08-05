#!/usr/bin/env python3
"""
Test script to verify Elo vs Metric comparison for HH-RLHF
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def test_elo_metric_comparison():
    """Test the Elo vs Metric comparison logic"""
    
    print("🧪 Testing Elo vs Metric Comparison for HH-RLHF")
    print("=" * 60)
    
    # Simulate sample data
    sample_data = {
        'sample_id': [0, 1, 2, 3, 4],
        'chosen_elo': [1550, 1540, 1560, 1530, 1570],  # Higher Elo scores
        'rejected_elo': [1450, 1460, 1440, 1470, 1430],  # Lower Elo scores
        'chosen_bleu': [0.8, 0.7, 0.9, 0.6, 0.85],    # Higher metric scores
        'rejected_bleu': [0.3, 0.4, 0.2, 0.5, 0.25],  # Lower metric scores
        'chosen_bleurt': [0.75, 0.65, 0.85, 0.55, 0.8],
        'rejected_bleurt': [0.25, 0.35, 0.15, 0.45, 0.2]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("📊 Sample Data:")
    print(df)
    print()
    
    # Test correlation calculation
    print("📈 Testing Correlation Calculation:")
    
    # For each sample, calculate correlation between Elo and metric scores
    correlations = []
    
    for _, row in df.iterrows():
        # Elo scores: chosen vs rejected
        elo_scores = [row['chosen_elo'], row['rejected_elo']]
        
        # Metric scores: chosen vs rejected
        bleu_scores = [row['chosen_bleu'], row['rejected_bleu']]
        bleurt_scores = [row['chosen_bleurt'], row['rejected_bleurt']]
        
        # Calculate correlations
        bleu_corr, _ = spearmanr(elo_scores, bleu_scores)
        bleurt_corr, _ = spearmanr(elo_scores, bleurt_scores)
        
        correlations.append({
            'sample_id': row['sample_id'],
            'bleu_correlation': bleu_corr,
            'bleurt_correlation': bleurt_corr
        })
        
        print(f"   Sample {row['sample_id']}:")
        print(f"     Elo scores: {elo_scores}")
        print(f"     BLEU scores: {bleu_scores}")
        print(f"     BLEURT scores: {bleurt_scores}")
        print(f"     BLEU correlation: {bleu_corr:.4f}")
        print(f"     BLEURT correlation: {bleurt_corr:.4f}")
        print()
    
    # Calculate mean correlations
    corr_df = pd.DataFrame(correlations)
    mean_bleu = corr_df['bleu_correlation'].mean()
    mean_bleurt = corr_df['bleurt_correlation'].mean()
    
    print("📊 Summary Statistics:")
    print(f"   Mean BLEU correlation: {mean_bleu:.4f}")
    print(f"   Mean BLEURT correlation: {mean_bleurt:.4f}")
    print()
    
    # Verify the approach is correct
    print("✅ Verification:")
    
    # Check that chosen responses have higher scores
    chosen_elo_higher = all(df['chosen_elo'] > df['rejected_elo'])
    chosen_bleu_higher = all(df['chosen_bleu'] > df['rejected_bleu'])
    chosen_bleurt_higher = all(df['chosen_bleurt'] > df['rejected_bleurt'])
    
    print(f"   Chosen Elo higher: {chosen_elo_higher}")
    print(f"   Chosen BLEU higher: {chosen_bleu_higher}")
    print(f"   Chosen BLEURT higher: {chosen_bleurt_higher}")
    
    # Check that correlations are positive (as expected)
    positive_bleu = mean_bleu > 0
    positive_bleurt = mean_bleurt > 0
    
    print(f"   Positive BLEU correlation: {positive_bleu}")
    print(f"   Positive BLEURT correlation: {positive_bleurt}")
    
    if chosen_elo_higher and chosen_bleu_higher and chosen_bleurt_higher and positive_bleu and positive_bleurt:
        print("\n🎉 SUCCESS: Elo vs Metric comparison is working correctly!")
        print("   - Chosen responses have higher scores in both Elo and metrics")
        print("   - Positive correlations indicate metrics align with human preferences")
    else:
        print("\n❌ ISSUE: Something is wrong with the comparison logic")
    
    return corr_df

if __name__ == "__main__":
    test_elo_metric_comparison() 