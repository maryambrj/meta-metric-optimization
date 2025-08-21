#!/usr/bin/env python3
"""
Test script for multi-dataset pipeline

This script tests the pipeline with both causal_relations and hh_rlhf datasets.
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔍 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr[:200]}...")
        return False

def test_causal_relations():
    """Test causal relations dataset"""
    print("🧪 Testing Causal Relations Dataset")
    print("=" * 50)
    
    # Test individual scripts
    tests = [
        ("python core_scripts/data_processing.py --dataset causal_relations", "Data processing"),
        ("python core_scripts/calc_metrics.py --dataset causal_relations", "Metric calculation"),
        ("python core_scripts/reg_test.py --dataset causal_relations", "Regression test"),
        ("python core_scripts/linear_regression_optimization.py --dataset causal_relations", "Optimization")
    ]
    
    for cmd, desc in tests:
        if not run_command(cmd, desc):
            return False
    
    return True

def test_hh_rlhf():
    """Test HH-RLHF dataset"""
    print("\n🧪 Testing HH-RLHF Dataset")
    print("=" * 50)
    
    # Test individual scripts
    tests = [
        ("python core_scripts/hh_rlhf_loader.py --num_samples 10", "HH-RLHF data loading"),
        ("python core_scripts/calc_metrics.py --dataset hh_rlhf", "Metric calculation"),
        ("python core_scripts/reg_test.py --dataset hh_rlhf", "Regression test"),
        ("python core_scripts/linear_regression_optimization.py --dataset hh_rlhf", "Optimization")
    ]
    
    for cmd, desc in tests:
        if not run_command(cmd, desc):
            return False
    
    return True

def test_pipeline():
    """Test full pipeline for both datasets"""
    print("\n🧪 Testing Full Pipeline")
    print("=" * 50)
    
    # Test pipeline commands
    tests = [
        ("python run_pipeline.py --step all --dataset causal_relations --skip-checks", "Full causal relations pipeline"),
        ("python run_pipeline.py --step all --dataset hh_rlhf --skip-checks", "Full HH-RLHF pipeline")
    ]
    
    for cmd, desc in tests:
        if not run_command(cmd, desc):
            return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Multi-Dataset Pipeline Test")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("run_pipeline.py"):
        print("❌ Please run this test from the project root directory")
        sys.exit(1)
    
    all_passed = True
    
    # Test configuration
    print("🔧 Testing configuration...")
    try:
        from config import get_dataset_config
        config_cr = get_dataset_config('causal_relations')
        config_hh = get_dataset_config('hh_rlhf')
        print("✅ Configuration loaded successfully")
        print(f"  - Causal Relations: {config_cr['num_samples']} samples, {config_cr['num_annotators']} annotators")
        print(f"  - HH-RLHF: {config_hh['num_samples']} samples, {config_hh['num_annotators']} annotators")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        all_passed = False
    
    # Run individual dataset tests
    if all_passed:
        all_passed &= test_causal_relations()
    
    if all_passed:
        all_passed &= test_hh_rlhf()
    
    # Test full pipeline
    if all_passed:
        all_passed &= test_pipeline()
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Multi-dataset pipeline is working correctly.")
        print("\n📊 You can now use:")
        print("  - python run_pipeline.py --step all --dataset causal_relations")
        print("  - python run_pipeline.py --step all --dataset hh_rlhf")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()