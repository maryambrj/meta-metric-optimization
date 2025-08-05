#!/usr/bin/env python3
"""
HH-RLHF Dataset Loader for Meta-Metric Optimization

This script loads and processes the Anthropic HH-RLHF dataset from Hugging Face.
The dataset contains pairwise comparisons with 'chosen' and 'rejected' responses.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import get_dataset_config, get_data_dir, get_annotations_dir, get_processed_data_dir

def download_hh_rlhf_dataset(subset='train', num_samples=1000):
    """
    Load the HH-RLHF dataset from Hugging Face using API calls
    
    Args:
        subset: 'train' or 'test'
        num_samples: Number of samples to load (None for all, but be careful with memory)
    
    Returns:
        pandas.DataFrame: Dataset with chosen and rejected columns
    """
    print(f"📥 Loading HH-RLHF dataset from Hugging Face API (subset: {subset})...")
    print(f"🔗 Dataset URL: https://huggingface.co/datasets/Anthropic/hh-rlhf")
    print(f"🌐 Using API calls - no local download required")
    
    try:
        # Use streaming=True for pure API access (no local caching)
        print(f"📡 Fetching samples via API (streaming)...")
        dataset = load_dataset("Anthropic/hh-rlhf", split=subset, streaming=True)
        
        data_list = []
        for i, example in enumerate(dataset):
            if num_samples is not None and i >= num_samples:
                break
            data_list.append(example)
        
        df = pd.DataFrame(data_list)
        
        print(f"📊 Successfully loaded {len(df)} samples via API")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📈 Available on Hugging Face: Train=161000 samples, Test=8550 samples")
        
        # Display sample data info
        if len(df) > 0:
            print(f"📝 Sample chosen response length: {len(df['chosen'].iloc[0])} characters")
            print(f"📝 Sample rejected response length: {len(df['rejected'].iloc[0])} characters")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset via API: {e}")
        print("💡 Troubleshooting:")
        print("   - Make sure you have 'datasets' package installed: pip install datasets")
        print("   - Check your internet connection")
        print("   - Try reducing num_samples if you're running out of memory")
        return None

def preprocess_hh_rlhf_data(df, dataset_name='hh_rlhf'):
    """
    Preprocess HH-RLHF data for evaluation
    
    Args:
        df: pandas.DataFrame with chosen and rejected columns
        dataset_name: Name of the dataset configuration
    
    Returns:
        dict: Processed data ready for evaluation
    """
    print("🔄 Preprocessing HH-RLHF data...")
    
    config = get_dataset_config(dataset_name)
    
    # Create directories
    os.makedirs(get_data_dir(dataset_name), exist_ok=True)
    os.makedirs(get_annotations_dir(dataset_name), exist_ok=True)
    os.makedirs(get_processed_data_dir(dataset_name), exist_ok=True)
    
    # Process data for pairwise comparison
    processed_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        # Extract chosen and rejected responses
        chosen_text = row['chosen'].strip()
        rejected_text = row['rejected'].strip()
        
        # Create sample data
        sample = {
            'sample_id': idx,
            'chosen_response': chosen_text,
            'rejected_response': rejected_text,
            'preference': 'chosen'  # In HH-RLHF, chosen is always preferred
        }
        
        processed_data.append(sample)
    
    # Save processed data
    processed_df = pd.DataFrame(processed_data)
    
    # Save main dataset file
    output_path = os.path.join(get_data_dir(dataset_name), "hh_rlhf_processed.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"💾 Saved processed data to: {output_path}")
    
    # Create annotation files for evaluation
    create_annotation_files(processed_df, dataset_name)
    
    # Create Elo-style rankings (chosen always wins)
    create_elo_rankings(processed_df, dataset_name)
    
    return processed_data

def create_annotation_files(df, dataset_name):
    """
    Create annotation files in the format expected by the evaluation pipeline
    """
    print("📝 Creating annotation files...")
    
    annotations_dir = get_annotations_dir(dataset_name)
    
    # Create chosen annotations
    chosen_annotations = []
    rejected_annotations = []
    
    for _, row in df.iterrows():
        # For chosen responses
        chosen_sample = {
            'sample_id': row['sample_id'],
            'text': row['chosen_response'],
            'quality_score': 1.0  # Chosen responses get higher score
        }
        chosen_annotations.append(chosen_sample)
        
        # For rejected responses
        rejected_sample = {
            'sample_id': row['sample_id'],
            'text': row['rejected_response'],
            'quality_score': 0.0  # Rejected responses get lower score
        }
        rejected_annotations.append(rejected_sample)
    
    # Save annotation files
    chosen_path = os.path.join(annotations_dir, "chosen.json")
    rejected_path = os.path.join(annotations_dir, "rejected.json")
    
    with open(chosen_path, 'w') as f:
        json.dump(chosen_annotations, f, indent=2)
    
    with open(rejected_path, 'w') as f:
        json.dump(rejected_annotations, f, indent=2)
    
    print(f"💾 Saved chosen annotations to: {chosen_path}")
    print(f"💾 Saved rejected annotations to: {rejected_path}")

def calculate_elo_rating(rating_a, rating_b, score_a, k_factor=32):
    """
    Calculate new Elo ratings after a match
    
    Args:
        rating_a: Current rating of player A
        rating_b: Current rating of player B
        score_a: Score for player A (1 for win, 0 for loss, 0.5 for draw)
        k_factor: K-factor for rating updates (default 32)
    
    Returns:
        tuple: (new_rating_a, new_rating_b)
    """
    # Expected score for player A
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    # Calculate new ratings
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - score_a) - (1 - expected_a))
    
    return new_rating_a, new_rating_b

def create_elo_rankings(df, dataset_name):
    """
    Create proper Elo rankings by simulating pairwise comparisons
    """
    print("🏆 Creating Elo rankings with proper algorithm...")
    
    data_dir = get_data_dir(dataset_name)
    
    # Initialize all responses with base Elo rating
    base_rating = 1500
    response_ratings = {}
    
    # Collect all unique responses and assign initial ratings
    for _, row in df.iterrows():
        chosen_text = row['chosen_response']
        rejected_text = row['rejected_response']
        
        if chosen_text not in response_ratings:
            response_ratings[chosen_text] = base_rating
        if rejected_text not in response_ratings:
            response_ratings[rejected_text] = base_rating
    
    print(f"📊 Initialized {len(response_ratings)} unique responses with Elo rating {base_rating}")
    
    # Simulate pairwise comparisons where chosen always beats rejected
    print("🔄 Simulating pairwise comparisons...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing comparisons"):
        chosen_text = row['chosen_response']
        rejected_text = row['rejected_response']
        
        # Get current ratings
        chosen_rating = response_ratings[chosen_text]
        rejected_rating = response_ratings[rejected_text]
        
        # Simulate match: chosen wins (score = 1 for chosen, 0 for rejected)
        new_chosen_rating, new_rejected_rating = calculate_elo_rating(
            chosen_rating, rejected_rating, score_a=1.0
        )
        
        # Update ratings
        response_ratings[chosen_text] = new_chosen_rating
        response_ratings[rejected_text] = new_rejected_rating
    
    # Create final Elo rankings DataFrame
    elo_data = []
    for _, row in df.iterrows():
        chosen_text = row['chosen_response']
        rejected_text = row['rejected_response']
        
        elo_row = {
            'sample_id': row['sample_id'],
            'chosen': response_ratings[chosen_text],
            'rejected': response_ratings[rejected_text],
            'winner': 'chosen',  # Chosen always wins
            'text_preview': row['chosen_response'][:100] + "..."
        }
        elo_data.append(elo_row)
    
    elo_df = pd.DataFrame(elo_data)
    
    # Save Elo rankings
    elo_path = os.path.join(data_dir, "final_elo_rankings.csv")
    elo_df.to_csv(elo_path, index=False)
    print(f"💾 Saved Elo rankings to: {elo_path}")
    
    # Print statistics
    chosen_ratings = [row['chosen'] for row in elo_data]
    rejected_ratings = [row['rejected'] for row in elo_data]
    
    print(f"📊 Elo Rating Statistics:")
    print(f"   Chosen responses: Mean={np.mean(chosen_ratings):.1f}, Std={np.std(chosen_ratings):.1f}")
    print(f"   Rejected responses: Mean={np.mean(rejected_ratings):.1f}, Std={np.std(rejected_ratings):.1f}")
    print(f"   Rating difference: Mean={np.mean(chosen_ratings) - np.mean(rejected_ratings):.1f}")
    
    # Create winner annotations file
    winner_data = []
    for _, row in df.iterrows():
        winner_row = {
            'sample_id': row['sample_id'],
            'text': row['chosen_response'][:200] + "...",
            'winner_annotation': json.dumps([{
                'chosen_text': row['chosen_response'],
                'rejected_text': row['rejected_response'],
                'preference': 'chosen'
            }])
        }
        winner_data.append(winner_row)
    
    winner_df = pd.DataFrame(winner_data)
    winner_path = os.path.join(data_dir, "winner_annotations.csv")
    winner_df.to_csv(winner_path, index=False)
    print(f"💾 Saved winner annotations to: {winner_path}")

def main():
    """Main function to load and process HH-RLHF dataset"""
    parser = argparse.ArgumentParser(description="Load and process HH-RLHF dataset from Hugging Face")
    parser.add_argument("--subset", choices=["train", "test"], default="train",
                       help="Dataset subset to load (train: 161k samples, test: 8.55k samples)")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to load (default: 1000, use larger for better statistics)")
    parser.add_argument("--dataset_name", default="hh_rlhf",
                       help="Dataset configuration name")
    
    args = parser.parse_args()
    
    print("🚀 HH-RLHF Dataset Loader")
    print("=" * 50)
    print(f"📊 Loading from Anthropic HH-RLHF dataset")
    print(f"🔗 Source: https://huggingface.co/datasets/Anthropic/hh-rlhf")
    print(f"📈 Available: Train=161k samples, Test=8.55k samples")
    print(f"⚙️  Requested: {args.num_samples} samples from {args.subset} split")
    
    # Download dataset
    df = download_hh_rlhf_dataset(args.subset, args.num_samples)
    
    if df is not None:
        # Preprocess data
        processed_data = preprocess_hh_rlhf_data(df, args.dataset_name)
        
        print("\n✅ HH-RLHF dataset processing complete!")
        print(f"📊 Processed {len(processed_data)} samples")
        print(f"📁 Data saved to: {get_data_dir(args.dataset_name)}")
        print("\nNext steps:")
        print("1. Run metric calculation: python run_pipeline.py --step metrics --dataset hh_rlhf")
        print("2. Run optimization: python run_pipeline.py --step optimization --dataset hh_rlhf")
    
    else:
        print("❌ Failed to download dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()