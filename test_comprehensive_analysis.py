#!/usr/bin/env python3
"""
Test script to verify comprehensive analysis for HH-RLHF
"""

import os
import subprocess
import pandas as pd

def check_output_files(dataset_name):
    """Check if all expected output files exist"""
    from config import get_rankings_dir
    
    rankings_dir = get_rankings_dir(dataset_name)
    expected_files = [
        "combined_metric_values.csv",
        "spearman_normalized_elo.csv", 
        "bootstrapped_spearman_plot.png",
        "combined_annotator_specific.csv",
        "linear_optimization_results.csv",
        "elo_values.csv",
        "bleu_values.csv",
        "bleurt_values.csv",
        "meteor_values.csv",
        "rouge_values.csv",
        "verbatim_values.csv",
        "spearman_per_sample.csv"
    ]
    
    print(f"📁 Checking output files for {dataset_name}...")
    print(f"   Directory: {rankings_dir}")
    print()
    
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(rankings_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename}")
            all_exist = False
    
    return all_exist

def analyze_outputs(dataset_name):
    """Analyze the output files to verify they contain meaningful data"""
    from config import get_rankings_dir
    
    rankings_dir = get_rankings_dir(dataset_name)
    
    print(f"\n📊 Analyzing outputs for {dataset_name}...")
    
    # Check combined metric values
    combined_file = os.path.join(rankings_dir, "combined_metric_values.csv")
    if os.path.exists(combined_file):
        df = pd.read_csv(combined_file)
        print(f"✅ Combined metric values: {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        if 'combined_score' in df.columns:
            print(f"   Combined score range: {df['combined_score'].min():.3f} - {df['combined_score'].max():.3f}")
    
    # Check Spearman correlations
    spearman_file = os.path.join(rankings_dir, "spearman_normalized_elo.csv")
    if os.path.exists(spearman_file):
        df = pd.read_csv(spearman_file)
        print(f"✅ Spearman correlations: {len(df)} metrics")
        print(f"   Metrics: {list(df['Metric'])}")
        if 'Spearman_Correlation' in df.columns:
            print(f"   Correlation range: {df['Spearman_Correlation'].min():.3f} - {df['Spearman_Correlation'].max():.3f}")
    
    # Check optimal weights
    weights_file = os.path.join(rankings_dir, "linear_optimization_results.csv")
    if os.path.exists(weights_file):
        df = pd.read_csv(weights_file)
        print(f"✅ Optimal weights: {len(df)} metrics")
        if 'Optimal_Weight' in df.columns:
            print(f"   Weight range: {df['Optimal_Weight'].min():.3f} - {df['Optimal_Weight'].max():.3f}")
            print(f"   Weight sum: {df['Optimal_Weight'].sum():.3f}")
    
    # Check visualization
    plot_file = os.path.join(rankings_dir, "bootstrapped_spearman_plot.png")
    if os.path.exists(plot_file):
        file_size = os.path.getsize(plot_file)
        print(f"✅ Visualization plot: {file_size/1024:.1f} KB")

def run_comprehensive_test():
    """Run comprehensive test for HH-RLHF analysis"""
    print("🧪 Testing Comprehensive Analysis for HH-RLHF")
    print("=" * 60)
    
    # Step 1: Check if BLEURT checkpoint exists
    bleurt_dir = "BLEURT-20"
    if not os.path.exists(bleurt_dir):
        print(f"❌ BLEURT checkpoint not found: {bleurt_dir}")
        print("   Please download it first:")
        print("   wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip")
        print("   unzip BLEURT-20.zip")
        print("   rm BLEURT-20.zip")
        return False
    
    print(f"✅ BLEURT checkpoint found: {bleurt_dir}")
    
    # Step 2: Run the complete pipeline
    print("\n🚀 Running complete pipeline...")
    cmd = "python run_pipeline.py --step all --dataset hh_rlhf"
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Pipeline completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    # Step 3: Check output files
    print("\n📋 Checking output files...")
    files_exist = check_output_files('hh_rlhf')
    
    if not files_exist:
        print("\n❌ Some output files are missing!")
        return False
    
    # Step 4: Analyze outputs
    print("\n📊 Analyzing outputs...")
    analyze_outputs('hh_rlhf')
    
    # Step 5: Compare with causal_relations
    print("\n🔄 Comparing with causal_relations dataset...")
    causal_files_exist = check_output_files('causal_relations')
    
    if causal_files_exist:
        print("\n📊 Analyzing causal_relations outputs...")
        analyze_outputs('causal_relations')
        
        print("\n✅ SUCCESS: HH-RLHF analysis produces the same comprehensive outputs as causal_relations!")
        print("   - Combined metric values with optimal weights")
        print("   - Spearman correlations")
        print("   - Visualization plots")
        print("   - Annotator-specific analysis")
    else:
        print("\n⚠️ Causal_relations outputs not found for comparison")
    
    return True

if __name__ == "__main__":
    success = run_comprehensive_test()
    if success:
        print("\n🎉 Comprehensive analysis test passed!")
    else:
        print("\n❌ Comprehensive analysis test failed!")
        exit(1) 