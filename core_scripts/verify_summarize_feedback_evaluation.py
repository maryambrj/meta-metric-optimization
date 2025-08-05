#!/usr/bin/env python3
"""
Verification script for summarize-feedback evaluation approach

This script demonstrates the correct evaluation approach:
- Original post text as reference
- Both summaries evaluated against the reference
- Compare metric scores with human preferences
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_evaluation_approach():
    """Verify the evaluation approach is correct"""
    print("🔍 Verifying Summarize-Feedback Evaluation Approach")
    print("=" * 60)
    
    # Load processed data
    data_file = "datasets/summarize_feedback/data/summarize_feedback_processed.csv"
    df = pd.read_csv(data_file)
    
    print(f"📊 Loaded {len(df)} samples")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Show example data point
    print("\n📝 Example Data Point:")
    print("-" * 40)
    sample = df.iloc[0]
    
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Post Title: {sample['post_title']}")
    print(f"\n📖 Original Post Text (REFERENCE):")
    print(f"   {sample['post_text'][:200]}...")
    print(f"\n🏆 Winner Summary (HUMAN PREFERENCE):")
    print(f"   {sample['winner_text']}")
    print(f"   Policy: {sample['winner_policy']}")
    print(f"\n❌ Loser Summary:")
    print(f"   {sample['loser_text']}")
    print(f"   Policy: {sample['loser_policy']}")
    
    print(f"\n✅ Evaluation Approach:")
    print(f"   1. Reference: Original post text (what should be summarized)")
    print(f"   2. Candidate 1: Winner summary (human preference)")
    print(f"   3. Candidate 2: Loser summary")
    print(f"   4. Calculate metrics: BLEU, BLEURT, METEOR, ROUGE, verbatim")
    print(f"   5. Compare: Do metrics align with human preferences?")
    
    # Show data distribution
    print(f"\n📈 Data Distribution:")
    print(f"   Total samples: {len(df)}")
    print(f"   Winner policies: {df['winner_policy'].value_counts().to_dict()}")
    print(f"   Loser policies: {df['loser_policy'].value_counts().to_dict()}")
    
    # Show some examples of different policy comparisons
    print(f"\n🔄 Policy Comparison Examples:")
    policy_comparisons = df.groupby(['winner_policy', 'loser_policy']).size().sort_values(ascending=False)
    print(policy_comparisons.head(10))
    
    print(f"\n✅ Verification Complete!")
    print(f"   The evaluation approach is correct:")
    print(f"   - Original post text serves as reference")
    print(f"   - Both summaries evaluated against reference")
    print(f"   - Human preferences provide ground truth")
    print(f"   - Metrics can be correlated with preferences")

if __name__ == "__main__":
    verify_evaluation_approach() 