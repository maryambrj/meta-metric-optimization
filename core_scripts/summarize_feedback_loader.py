#!/usr/bin/env python3
"""
Summarize-from-Feedback Dataset Loader for Meta-Metric Optimization

This script loads and processes the OpenAI summarize-from-feedback dataset.
The dataset contains pairwise comparisons between summaries with human preferences.
"""

import os
import sys
import pandas as pd
import json
import requests
from tqdm import tqdm
import argparse
import gzip
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import get_dataset_config, get_data_dir, get_annotations_dir, get_processed_data_dir

def download_summarize_feedback_dataset(split='train', num_samples=1000):
    """
    Download the summarize-from-feedback dataset from Azure Blob Storage
    
    Args:
        split: 'train', 'valid1', or 'valid2'
        num_samples: Number of samples to load (None for all)
    
    Returns:
        pandas.DataFrame: Dataset with comparison data
    """
    print(f"📥 Downloading summarize-from-feedback dataset (split: {split})...")
    print(f"🔗 Dataset URL: https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/")
    
    # Dataset URLs for different splits
    base_url = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/comparisons"
    
    # For training data, we use batches 3-10
    if split == 'train':
        batch_files = [f"batch{i}.json" for i in range(3, 11)]
    elif split == 'valid1':
        batch_files = ["batch1.json", "batch2.json"]
    elif split == 'valid2':
        batch_files = ["batch11.json", "batch12.json"]
    else:
        raise ValueError(f"Invalid split: {split}. Use 'train', 'valid1', or 'valid2'")
    
    all_data = []
    
    for batch_file in batch_files:
        url = f"{base_url}/{batch_file}"
        print(f"📡 Downloading {batch_file}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            batch_data = []
            for line_num, line in enumerate(response.iter_lines()):
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        batch_data.append(data)
                        
                        # Limit samples if specified
                        if num_samples is not None and len(batch_data) >= num_samples:
                            break
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping malformed line {line_num}: {e}")
                        continue
            
            all_data.extend(batch_data)
            print(f"✅ Downloaded {len(batch_data)} samples from {batch_file}")
            
        except requests.RequestException as e:
            print(f"❌ Error downloading {batch_file}: {e}")
            continue
    
    if not all_data:
        print("❌ No data downloaded. Check your internet connection and try again.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    print(f"📊 Successfully loaded {len(df)} total samples")
    print(f"📋 Columns: {list(df.columns)}")
    
    return df

def preprocess_summarize_feedback_data(df, dataset_name='summarize_feedback'):
    """
    Preprocess summarize-from-feedback data for evaluation
    
    Args:
        df: pandas.DataFrame with comparison data
        dataset_name: Name of the dataset configuration
    
    Returns:
        dict: Processed data ready for evaluation
    """
    print("🔄 Preprocessing summarize-from-feedback data...")
    
    config = get_dataset_config(dataset_name)
    
    # Create directories
    os.makedirs(get_data_dir(dataset_name), exist_ok=True)
    os.makedirs(get_annotations_dir(dataset_name), exist_ok=True)
    os.makedirs(get_processed_data_dir(dataset_name), exist_ok=True)
    
    # Process data for pairwise comparison
    processed_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        # Extract post information
        post_info = row.get('info', {})
        post_text = post_info.get('post', '')
        post_title = post_info.get('title', '')
        
        # Extract summaries
        summaries = row.get('summaries', [])
        if len(summaries) < 2:
            continue
        
        summary1 = summaries[0]
        summary2 = summaries[1]
        
        # Get choice (0 for first summary, 1 for second)
        choice = row.get('choice', 0)
        
        # Determine winner and loser
        if choice == 0:
            winner_text = summary1.get('text', '')
            loser_text = summary2.get('text', '')
            winner_policy = summary1.get('policy', 'unknown')
            loser_policy = summary2.get('policy', 'unknown')
        else:
            winner_text = summary2.get('text', '')
            loser_text = summary1.get('text', '')
            winner_policy = summary2.get('policy', 'unknown')
            loser_policy = summary1.get('policy', 'unknown')
        
        # Create sample data
        sample = {
            'sample_id': idx,
            'post_text': post_text,
            'post_title': post_title,
            'winner_text': winner_text,
            'loser_text': loser_text,
            'winner_policy': winner_policy,
            'loser_policy': loser_policy,
            'preference': 'winner'  # winner is always preferred
        }
        
        processed_data.append(sample)
    
    # Save processed data
    processed_df = pd.DataFrame(processed_data)
    
    # Save main dataset file
    output_path = os.path.join(get_data_dir(dataset_name), "summarize_feedback_processed.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"💾 Saved processed data to: {output_path}")
    
    # Create annotation files for evaluation
    create_annotation_files(processed_df, dataset_name)
    
    # Create simple winner/loser data for pairwise logistic regression (no Elo needed)
    create_pairwise_data(processed_df, dataset_name)
    
    return processed_data

def create_annotation_files(df, dataset_name):
    """
    Create annotation files in the format expected by the evaluation pipeline
    """
    print("📝 Creating annotation files...")
    
    annotations_dir = get_annotations_dir(dataset_name)
    
    # Create winner and loser annotations
    winner_annotations = []
    loser_annotations = []
    
    for _, row in df.iterrows():
        # For winner summaries
        winner_sample = {
            'sample_id': row['sample_id'],
            'text': row['winner_text'],
            'quality_score': 1.0,  # Winner summaries get higher score
            'policy': row['winner_policy']
        }
        winner_annotations.append(winner_sample)
        
        # For loser summaries
        loser_sample = {
            'sample_id': row['sample_id'],
            'text': row['loser_text'],
            'quality_score': 0.0,  # Loser summaries get lower score
            'policy': row['loser_policy']
        }
        loser_annotations.append(loser_sample)
    
    # Save annotation files
    winner_path = os.path.join(annotations_dir, "winner.json")
    loser_path = os.path.join(annotations_dir, "loser.json")
    
    with open(winner_path, 'w') as f:
        json.dump(winner_annotations, f, indent=2)
    
    with open(loser_path, 'w') as f:
        json.dump(loser_annotations, f, indent=2)
    
    print(f"💾 Saved winner annotations to: {winner_path}")
    print(f"💾 Saved loser annotations to: {loser_path}")

def create_pairwise_data(df, dataset_name):
    """
    Create pairwise data for logistic regression (no Elo needed)
    """
    print("📊 Creating pairwise preference data...")
    
    data_dir = get_data_dir(dataset_name)
    
    # Create pairwise comparison data
    pairwise_data = []
    for _, row in df.iterrows():
        pairwise_row = {
            'sample_id': row['sample_id'],
            'winner': 1,  # Winner always wins (binary preference)
            'loser': 0,   # Loser always loses
            'winner_policy': row['winner_policy'],
            'loser_policy': row['loser_policy'],
            'winner_text': row['winner_text'][:100] + "..." if len(row['winner_text']) > 100 else row['winner_text'],
            'loser_text': row['loser_text'][:100] + "..." if len(row['loser_text']) > 100 else row['loser_text']
        }
        pairwise_data.append(pairwise_row)
    
    pairwise_df = pd.DataFrame(pairwise_data)
    
    # Save pairwise data (compatible with existing pipeline)
    pairwise_path = os.path.join(data_dir, "final_elo_rankings.csv")
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"💾 Saved pairwise preference data to: {pairwise_path}")
    
    print(f"📊 Pairwise Data Statistics:")
    print(f"   Total samples: {len(pairwise_df)}")
    print(f"   Winner preference: 100% (by definition)")
    print(f"   Unique winner policies: {pairwise_df['winner_policy'].nunique()}")
    print(f"   Unique loser policies: {pairwise_df['loser_policy'].nunique()}")
    
    # Create winner annotations file (for compatibility)
    winner_data = []
    for _, row in df.iterrows():
        winner_row = {
            'sample_id': row['sample_id'],
            'text': row['winner_text'][:200] + "..." if len(row['winner_text']) > 200 else row['winner_text'],
            'winner_annotation': json.dumps([{
                'winner_text': row['winner_text'],
                'loser_text': row['loser_text'],
                'preference': 'winner'
            }])
        }
        winner_data.append(winner_row)
    
    winner_df = pd.DataFrame(winner_data)
    winner_path = os.path.join(data_dir, "winner_annotations.csv")
    winner_df.to_csv(winner_path, index=False)
    print(f"💾 Saved winner annotations to: {winner_path}")

def main():
    """Main function to load and process summarize-from-feedback dataset"""
    parser = argparse.ArgumentParser(description="Load and process summarize-from-feedback dataset")
    parser.add_argument("--split", choices=["train", "valid1", "valid2"], default="train",
                       help="Dataset split to load")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to load (default: 1000)")
    parser.add_argument("--dataset_name", default="summarize_feedback",
                       help="Dataset configuration name")
    
    args = parser.parse_args()
    
    print("🚀 Summarize-from-Feedback Dataset Loader")
    print("=" * 50)
    print(f"📊 Loading from OpenAI summarize-from-feedback dataset")
    print(f"🔗 Source: https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/")
    print(f"📈 Split: {args.split}")
    print(f"⚙️  Requested: {args.num_samples} samples")
    
    # Download dataset
    df = download_summarize_feedback_dataset(args.split, args.num_samples)
    
    if df is not None:
        # Preprocess data
        processed_data = preprocess_summarize_feedback_data(df, args.dataset_name)
        
        print("\n✅ Summarize-from-feedback dataset processing complete!")
        print(f"📊 Processed {len(processed_data)} samples")
        print(f"📁 Data saved to: {get_data_dir(args.dataset_name)}")
        print("\nNext steps:")
        print("1. Run metric calculation: python run_pipeline.py --step metrics --dataset summarize_feedback")
        print("2. Run optimization: python run_pipeline.py --step optimization --dataset summarize_feedback")
    
    else:
        print("❌ Failed to download dataset")
        sys.exit(1)

if __name__ == "__main__":
    main() 