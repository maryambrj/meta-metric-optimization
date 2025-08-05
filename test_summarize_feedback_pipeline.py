#!/usr/bin/env python3
"""
Test script for the summarize-from-feedback pipeline

This script tests the complete pipeline for the summarize-from-feedback dataset
with a small sample to ensure everything works correctly.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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

def test_summarize_feedback_pipeline():
    """Test the complete summarize-feedback pipeline"""
    print("🧪 Testing Summarize-from-Feedback Pipeline")
    print("=" * 60)
    
    # Test 1: Data loading with small sample
    print("\n📥 Test 1: Data Loading")
    cmd = f"cd '{project_root}/core_scripts' && python summarize_feedback_loader.py --split train --num_samples 10 --dataset_name summarize_feedback"
    if not run_command(cmd, "Loading summarize-feedback dataset (10 samples)"):
        return False
    
    # Test 2: Metric calculation
    print("\n📊 Test 2: Metric Calculation")
    cmd = f"cd '{project_root}/core_scripts' && python calc_metrics.py --dataset summarize_feedback --batch_size 8"
    if not run_command(cmd, "Calculating metrics for summarize-feedback"):
        return False
    
    # Test 3: Check output files
    print("\n📁 Test 3: Output File Verification")
    from config import get_data_dir, get_rankings_dir
    
    data_dir = get_data_dir('summarize_feedback')
    rankings_dir = get_rankings_dir('summarize_feedback')
    
    required_files = [
        os.path.join(data_dir, "summarize_feedback_processed.csv"),
        os.path.join(data_dir, "final_elo_rankings.csv"),
        os.path.join(data_dir, "detailed_scores.csv"),
        os.path.join(data_dir, "winner_annotations.csv")
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {os.path.basename(file_path)}")
        else:
            print(f"❌ Missing: {os.path.basename(file_path)}")
            return False
    
    # Test 4: Run full pipeline
    print("\n🚀 Test 4: Full Pipeline")
    cmd = f"cd '{project_root}' && python run_pipeline.py --step all --dataset summarize_feedback --skip-checks"
    if not run_command(cmd, "Running complete pipeline"):
        return False
    
    print("\n🎉 All tests passed! Summarize-feedback pipeline is working correctly.")
    return True

def main():
    """Main test function"""
    print("🧪 Summarize-from-Feedback Pipeline Test Suite")
    print("=" * 60)
    
    success = test_summarize_feedback_pipeline()
    
    if success:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. Run with more data: python core_scripts/summarize_feedback_loader.py --num_samples 1000")
        print("2. Run full pipeline: python run_pipeline.py --dataset summarize_feedback")
        print("3. Check results in datasets/summarize_feedback/rankings/")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 