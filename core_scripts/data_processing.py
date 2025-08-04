#!/usr/bin/env python3
"""
Data Processing Script for Meta-Metric Optimization

This script processes annotation data and creates winner annotations CSV.
"""

import pandas as pd
import json
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import *

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process annotation data")
    parser.add_argument("--dataset", default="causal_relations", help="Dataset name")
    args = parser.parse_args()
    
    print(f"🔄 Starting data processing for {args.dataset}...")
    
    # Get dataset-specific paths
    from config import get_dataset_config, get_data_dir, get_annotations_dir
    config = get_dataset_config(args.dataset)
    data_dir = get_data_dir(args.dataset)
    annotations_dir = get_annotations_dir(args.dataset)
    
    # === CONFIGURATION ===
    annotation_dir = annotations_dir  # folder with annotation JSON files
    winner_csv_path = os.path.join(data_dir, "final_elo_rankings.csv")  # CSV with the 'winner' column
    output_csv_path = os.path.join(data_dir, "winner_annotations.csv")  # output path
    
    print(f"📂 Annotation directory: {annotation_dir}")
    print(f"📊 Winner CSV path: {winner_csv_path}")
    print(f"💾 Output CSV path: {output_csv_path}")
    
    # === MAPPING: CSV winner label -> actual JSON file name (no extension) ===
    replacements = {
        "Aadarsh": "Human1",
        "Ashlin": "Human2",
        "Kuldeep": "Human3",
        "Maryam": "Human4",
        "Nate": "Human5",
        "Riley": "Human6",
        "Spencer": "Human7"
    }
    # === Add LLMs explicitly ===
    replacements.update({
        "llama2_model": "llama2",
        "llama3_model": "llama3",
        "mistral_model": "mistral"
    })
    
    # Invert the mapping: CSV label (e.g., Human1) -> JSON file base name (e.g., Aadarsh)
    csv_to_json_labeler = {v: k for k, v in replacements.items()}
    
    # === 1. Load all JSON annotation files ===
    print("📋 Loading annotation files...")
    annotations = {}
    
    # Check if annotation directory exists
    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
    
    # Load human annotations
    human_dir = os.path.join(annotation_dir, "humans")
    if os.path.exists(human_dir):
        for file in os.listdir(human_dir):
            if file.endswith('.json'):
                labeler = file.replace('.json', '')
                file_path = os.path.join(human_dir, file)
                with open(file_path, 'r') as f:
                    annotations[labeler] = json.load(f)
                print(f"  ✅ Loaded {labeler}: {len(annotations[labeler])} samples")
    
    # Load LLM annotations
    llm_dir = os.path.join(annotation_dir, "LLMs")
    if os.path.exists(llm_dir):
        for file in os.listdir(llm_dir):
            if file.endswith('.json'):
                labeler = file.replace('.json', '')
                file_path = os.path.join(llm_dir, file)
                with open(file_path, 'r') as f:
                    annotations[labeler] = json.load(f)
                print(f"  ✅ Loaded {labeler}: {len(annotations[labeler])} samples")
    
    print(f"📋 Total annotators loaded: {len(annotations)}")
    
    # === 2. Load winner info from CSV ===
    print("📊 Loading winner CSV...")
    if not os.path.exists(winner_csv_path):
        raise FileNotFoundError(f"Winner CSV not found: {winner_csv_path}")
    
    elo_df = pd.read_csv(winner_csv_path)
    
    if 'winner' not in elo_df.columns or 'sentence_start' not in elo_df.columns:
        raise ValueError("CSV must contain 'winner' and 'sentence_start' columns.")
    
    print(f"📊 Loaded {len(elo_df)} rows from winner CSV")
    
    # === 3. Build output rows ===
    print("🔗 Building output rows...")
    output_rows = []
    
    for i in range(len(elo_df)):
        csv_label = elo_df.loc[i, 'winner']
        sentence = elo_df.loc[i, 'sentence_start']
        
        if csv_label not in csv_to_json_labeler:
            raise ValueError(f"CSV label '{csv_label}' not found in mapping.")
        
        json_labeler = csv_to_json_labeler[csv_label]
        
        if json_labeler not in annotations:
            raise ValueError(f"Annotation file for labeler '{json_labeler}' not loaded.")
        
        try:
            winner_annotation = annotations[json_labeler][i]['causal relations']
        except IndexError:
            raise ValueError(f"Index {i} out of range for labeler '{json_labeler}'.")
        
        output_rows.append({
            'sample_id': i + 1,
            'text': sentence,
            'winner_annotation': json.dumps(winner_annotation)
        })
    
    # === 4. Save final CSV ===
    print("💾 Saving output CSV...")
    output_df = pd.DataFrame(output_rows)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"✅ Winner annotations saved to: {output_csv_path}")
    
    # === 5. Normalize directions in annotation files ===
    print("🔄 Normalizing causal relation directions...")
    
    # === Conversion mapping: normalize all directions ===
    direction_map = {
        "positive": "increase",
        "negative": "decrease",
        "Increase": "increase",
        "increase": "increase",
        "Decrease": "decrease",
        "decrease": "decrease"
    }
    
    # === Function to normalize causal relations ===
    def normalize_directions(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        
        for item in data:
            for rel in item.get("causal relations", []):
                dir_orig = rel["direction"]
                rel["direction"] = direction_map.get(dir_orig.strip(), dir_orig.strip())
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    # === Process LLM files ===
    if os.path.exists(llm_dir):
        for file in os.listdir(llm_dir):
            if file.endswith(".json"):
                file_path = os.path.join(llm_dir, file)
                normalize_directions(file_path)
                print(f"  ✅ Normalized {file}")
    
    # === Process human files ===
    if os.path.exists(human_dir):
        for file in os.listdir(human_dir):
            if file.endswith(".json"):
                file_path = os.path.join(human_dir, file)
                normalize_directions(file_path)
                print(f"  ✅ Normalized {file}")
    
    print("✅ Direction normalization complete!")
    
    # === 6. Create combined JSON files for each annotator ===
    print("🔗 Creating combined JSON files...")
    
    # Load winner annotations
    winner_annotations = []
    with open(output_csv_path, 'r') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            winner_annotations.append({
                'sample_id': int(row['sample_id']),
                'text': row['text'],
                'winner_annotation': json.loads(row['winner_annotation'])
            })
    
    # Process each annotator
    filenames = {}
    for labeler in annotations.keys():
        print(f"  🔗 Processing {labeler}...")
        
        # Create combined data structure
        combined_json = {
            "train": winner_annotations,
            "test": annotations[labeler]
        }
        
        # Save to processed_data directory
        out_file = os.path.join(PROCESSED_DATA_DIR, f"{labeler}_elo_full.json")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        with open(out_file, "w") as f:
            json.dump(combined_json, f, indent=2)
        
        filenames[labeler] = out_file  # Save for future use
        print(f"  ✅ Saved {out_file}")
    
    print("✅ Data processing complete!")
    print(f"📊 Processed {len(annotations)} annotators")
    print(f"📋 Created {len(output_rows)} winner annotation rows")
    print(f"🗂️ Saved combined files to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()