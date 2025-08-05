#!/usr/bin/env python3
"""
Main pipeline script for Meta-Metric Optimization

This script orchestrates the entire evaluation pipeline:
1. Data preprocessing
2. Metric calculation
3. Meta-metric optimization
4. Results analysis
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import *
from config import get_dataset_config, get_data_dir, get_annotations_dir, get_processed_data_dir, get_rankings_dir

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}:")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def check_dependencies(dataset_name='causal_relations'):
    """Check if required files and directories exist"""
    print(f"Checking dependencies for dataset: {dataset_name}...")
    
    config = get_dataset_config(dataset_name)
    data_dir = get_data_dir(dataset_name)
    annotations_dir = get_annotations_dir(dataset_name)
    processed_data_dir = get_processed_data_dir(dataset_name)
    rankings_dir = get_rankings_dir(dataset_name)
    
    required_dirs = [data_dir, annotations_dir, processed_data_dir, rankings_dir]
    
    if dataset_name in ['hh_rlhf', 'summarize_feedback']:
        # For HH-RLHF and summarize-feedback, create directories if they don't exist
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Ensured directory exists: {dir_path}")
    else:
        # For causal_relations, check if directories exist
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"❌ Missing directory: {dir_path}")
                return False
            else:
                print(f"✅ Found directory: {dir_path}")
    
    # Check for dataset-specific files
    if dataset_name == 'causal_relations':
        required_files = [
            os.path.join(data_dir, "winner_annotations.csv"),
            os.path.join(data_dir, "final_elo_rankings.csv")
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Missing file: {file_path}")
                return False
            else:
                print(f"✅ Found file: {file_path}")
    
    return True

def run_data_processing(dataset_name='causal_relations'):
    """Run data preprocessing step"""
    if dataset_name == 'hh_rlhf':
        # For HH-RLHF, run the dataset loader
        script_path = os.path.join(CORE_SCRIPTS_DIR, "hh_rlhf_loader.py")
        if not os.path.exists(script_path):
            print(f"❌ HH-RLHF loader script not found: {script_path}")
            return False
        cmd = f"cd '{CORE_SCRIPTS_DIR}' && python hh_rlhf_loader.py --dataset_name {dataset_name}"
        return run_command(cmd, f"Loading {dataset_name} dataset")
    elif dataset_name == 'summarize_feedback':
        # For summarize-feedback, run the dataset loader
        script_path = os.path.join(CORE_SCRIPTS_DIR, "summarize_feedback_loader.py")
        if not os.path.exists(script_path):
            print(f"❌ Summarize-feedback loader script not found: {script_path}")
            return False
        cmd = f"cd '{CORE_SCRIPTS_DIR}' && python summarize_feedback_loader.py --dataset_name {dataset_name}"
        return run_command(cmd, f"Loading {dataset_name} dataset")
    else:
        # For causal_relations, run the original data processing
        script_path = os.path.join(CORE_SCRIPTS_DIR, "data_processing.py")
        if not os.path.exists(script_path):
            print(f"❌ Data processing script not found: {script_path}")
            return False
        cmd = f"cd '{CORE_SCRIPTS_DIR}' && python data_processing.py"
        return run_command(cmd, "Running data preprocessing")

def run_metric_calculation(dataset_name='causal_relations'):
    """Run metric calculation step"""
    script_path = os.path.join(CORE_SCRIPTS_DIR, "calc_metrics.py")
    if not os.path.exists(script_path):
        print(f"❌ Metric calculation script not found: {script_path}")
        return False
    
    cmd = f"cd '{CORE_SCRIPTS_DIR}' && python calc_metrics.py --dataset {dataset_name}"
    if not run_command(cmd, f"Running metric calculation for {dataset_name}"):
        return False
    
    # Also run regression test to create ranking tables
    script_path = os.path.join(CORE_SCRIPTS_DIR, "reg_test.py")
    if not os.path.exists(script_path):
        print(f"❌ Regression test script not found: {script_path}")
        return False
    
    cmd = f"cd '{CORE_SCRIPTS_DIR}' && python reg_test.py --dataset {dataset_name}"
    return run_command(cmd, f"Running regression test for {dataset_name}")

def run_meta_metric_optimization(dataset_name='causal_relations'):
    """Run meta-metric optimization step"""
    script_path = os.path.join(CORE_SCRIPTS_DIR, "linear_optimization.py")
    if not os.path.exists(script_path):
        print(f"❌ Linear optimization script not found: {script_path}")
        return False
    
    cmd = f"cd '{CORE_SCRIPTS_DIR}' && python linear_optimization.py --dataset {dataset_name}"
    return run_command(cmd, f"Running linear combination optimization for {dataset_name}")

def generate_report():
    """Generate a summary report"""
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*60}")
    
    # Check if key output files exist
    output_files = [
        os.path.join(RANKINGS_DIR, "combined_metric_values.csv"),
        os.path.join(RANKINGS_DIR, "spearman_normalized_elo.csv"),
        os.path.join(RANKINGS_DIR, "bootstrapped_spearman_plot.png")
    ]
    
    print("\nOutput files generated:")
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"✅ {os.path.basename(file_path)}")
        else:
            print(f"❌ {os.path.basename(file_path)}")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Check the rankings/ directory for metric scores")
    print("2. Review the combined_metric_values.csv for optimized weights")
    print("3. Examine bootstrapped_spearman_plot.png for correlation analysis")
    print("4. Run individual notebooks for detailed analysis")

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description="Meta-Metric Optimization Pipeline")
    parser.add_argument("--step", choices=["data", "metrics", "optimization", "all"], 
                       default="all", help="Which step to run")
    parser.add_argument("--dataset", choices=["causal_relations", "hh_rlhf", "summarize_feedback"], 
                       default="causal_relations", help="Which dataset to use")
    parser.add_argument("--skip-checks", action="store_true", 
                       help="Skip dependency checks")
    
    args = parser.parse_args()
    
    print("🚀 Meta-Metric Optimization and Auto-Elo Ranking Pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {args.dataset}")
    
    if not args.skip_checks:
        if not check_dependencies(args.dataset):
            print("❌ Dependency check failed. Please ensure all required files exist.")
            sys.exit(1)
    
    success = True
    
    if args.step in ["data", "all"]:
        success &= run_data_processing(args.dataset)
    
    if args.step in ["metrics", "all"] and success:
        success &= run_metric_calculation(args.dataset)
    
    if args.step in ["optimization", "all"] and success:
        success &= run_meta_metric_optimization(args.dataset)
    
    if success:
        generate_report()
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 