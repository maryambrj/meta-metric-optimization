#!/usr/bin/env python3
"""
Test script for HH-RLHF pipeline

This script tests the complete pipeline:
1. Download BLEURT checkpoint
2. Run metrics calculation
3. Run regression test
4. Run linear optimization
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def check_bleurt_checkpoint():
    """Check if BLEURT checkpoint exists"""
    checkpoint_dir = "BLEURT-20"
    if os.path.exists(checkpoint_dir):
        print(f"✅ BLEURT checkpoint found: {checkpoint_dir}")
        return True
    else:
        print(f"❌ BLEURT checkpoint not found: {checkpoint_dir}")
        print("   Please download it first:")
        print("   wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip")
        print("   unzip BLEURT-20.zip")
        print("   rm BLEURT-20.zip")
        return False

def test_metrics_calculation():
    """Test metrics calculation"""
    return run_command(
        "python run_pipeline.py --step metrics --dataset hh_rlhf",
        "Metrics calculation for HH-RLHF"
    )

def test_regression():
    """Test regression analysis"""
    return run_command(
        "python run_pipeline.py --step regression --dataset hh_rlhf",
        "Regression analysis for HH-RLHF"
    )

def test_linear_optimization():
    """Test linear optimization"""
    return run_command(
        "python core_scripts/linear_optimization.py --dataset hh_rlhf",
        "Linear combination optimization for HH-RLHF"
    )

def check_output_files():
    """Check if output files were created"""
    print(f"\n{'='*60}")
    print("Checking output files...")
    print(f"{'='*60}")
    
    expected_files = [
        "datasets/hh_rlhf/data/detailed_scores.csv",
        "datasets/hh_rlhf/rankings/elo_values.csv",
        "datasets/hh_rlhf/rankings/bleu_values.csv",
        "datasets/hh_rlhf/rankings/bleurt_values.csv",
        "datasets/hh_rlhf/rankings/meteor_values.csv",
        "datasets/hh_rlhf/rankings/rouge_values.csv",
        "datasets/hh_rlhf/rankings/verbatim_values.csv",
        "datasets/hh_rlhf/rankings/spearman_per_sample.csv",
        "datasets/hh_rlhf/rankings/linear_optimization_results.csv"
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Main test function"""
    print("🧪 Testing HH-RLHF Pipeline")
    print("=" * 60)
    
    # Check BLEURT checkpoint
    if not check_bleurt_checkpoint():
        print("\n❌ Please download BLEURT checkpoint first")
        return False
    
    # Test metrics calculation
    if not test_metrics_calculation():
        print("\n❌ Metrics calculation failed")
        return False
    
    # Test regression
    if not test_regression():
        print("\n❌ Regression analysis failed")
        return False
    
    # Test linear optimization
    if not test_linear_optimization():
        print("\n❌ Linear optimization failed")
        return False
    
    # Check output files
    if not check_output_files():
        print("\n❌ Some output files are missing")
        return False
    
    print(f"\n{'='*60}")
    print("🎉 All tests passed! HH-RLHF pipeline is working correctly.")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 