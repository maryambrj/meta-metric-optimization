#!/usr/bin/env python3
"""
Test script for the dual-method optimization pipeline

This script tests both the correlation-based optimization (for causal data) 
and pairwise logistic regression (for summary data) implementations.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_dataset_config, get_data_dir, get_annotations_dir, get_rankings_dir

def create_mock_summary_data(num_samples=50):
    """
    Create mock summary data for testing pairwise logistic regression
    """
    print("🔧 Creating mock summary data...")
    
    # Create mock data similar to summarize_feedback format
    mock_data = []
    
    for i in range(num_samples):
        # Create two summaries with different quality
        winner_quality = np.random.uniform(0.7, 1.0)
        loser_quality = np.random.uniform(0.3, 0.6)
        
        sample = {
            'sample_id': i,
            'winner_text': f"High quality summary {i} with good content and structure.",
            'loser_text': f"Lower quality summary {i} with issues.",
            'winner_policy': 'model_a',
            'loser_policy': 'model_b',
            'winner_quality': winner_quality,
            'loser_quality': loser_quality
        }
        mock_data.append(sample)
    
    # Create DataFrame
    df = pd.DataFrame(mock_data)
    
    # Save to test directory
    test_dir = "/tmp/test_summarize_feedback"
    os.makedirs(test_dir, exist_ok=True)
    
    data_dir = os.path.join(test_dir, "data")
    annotations_dir = os.path.join(test_dir, "annotations") 
    rankings_dir = os.path.join(test_dir, "rankings")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(rankings_dir, exist_ok=True)
    
    # Save processed data
    df.to_csv(os.path.join(data_dir, "summarize_feedback_processed.csv"), index=False)
    
    # Create Elo rankings (winner always has higher Elo)
    elo_data = []
    for _, row in df.iterrows():
        base_elo = 1500
        winner_boost = np.random.uniform(50, 200)
        loser_penalty = np.random.uniform(50, 200)
        
        elo_row = {
            'sample_id': row['sample_id'],
            'winner': base_elo + winner_boost,
            'loser': base_elo - loser_penalty,
            'winner_policy': row['winner_policy'],
            'loser_policy': row['loser_policy']
        }
        elo_data.append(elo_row)
    
    elo_df = pd.DataFrame(elo_data)
    elo_df.to_csv(os.path.join(data_dir, "final_elo_rankings.csv"), index=False)
    
    # Create mock metric files
    metrics = ['bleu', 'meteor', 'rouge', 'bleurt']
    
    for metric in metrics:
        metric_data = []
        
        for _, row in df.iterrows():
            # Winner should generally have better metrics
            if metric == 'bleu':
                winner_score = row['winner_quality'] * 0.8 + np.random.normal(0, 0.1)
                loser_score = row['loser_quality'] * 0.8 + np.random.normal(0, 0.1)
            elif metric == 'meteor':
                winner_score = row['winner_quality'] * 0.9 + np.random.normal(0, 0.1) 
                loser_score = row['loser_quality'] * 0.9 + np.random.normal(0, 0.1)
            elif metric == 'rouge':
                winner_score = row['winner_quality'] * 0.85 + np.random.normal(0, 0.1)
                loser_score = row['loser_quality'] * 0.85 + np.random.normal(0, 0.1)
            else:  # bleurt
                winner_score = (row['winner_quality'] - 0.5) * 2 + np.random.normal(0, 0.2)  # Scale to [-1, 1]
                loser_score = (row['loser_quality'] - 0.5) * 2 + np.random.normal(0, 0.2)
            
            # Clip scores to reasonable ranges
            winner_score = np.clip(winner_score, 0, 1) if metric != 'bleurt' else np.clip(winner_score, -1, 1)
            loser_score = np.clip(loser_score, 0, 1) if metric != 'bleurt' else np.clip(loser_score, -1, 1)
            
            metric_data.append({
                'sample_id': row['sample_id'],
                'winner': winner_score,
                'loser': loser_score
            })
        
        metric_df = pd.DataFrame(metric_data)
        metric_df.set_index('sample_id', inplace=True)
        metric_df.to_csv(os.path.join(rankings_dir, f"{metric}_values.csv"))
    
    print(f"✅ Created mock summary data in {test_dir}")
    print(f"   Samples: {num_samples}")
    print(f"   Metrics: {metrics}")
    
    return test_dir

def create_mock_causal_data(num_samples=30, num_annotators=10):
    """
    Create mock causal relation data for testing correlation-based optimization
    """
    print("🔧 Creating mock causal data...")
    
    test_dir = "/tmp/test_causal_relations"
    os.makedirs(test_dir, exist_ok=True)
    
    data_dir = os.path.join(test_dir, "data")
    annotations_dir = os.path.join(test_dir, "annotations")
    rankings_dir = os.path.join(test_dir, "rankings")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(rankings_dir, exist_ok=True)
    
    # Create Elo rankings for all annotators
    elo_data = {}
    annotators = [f"Annotator_{i}" for i in range(num_annotators)]
    
    for sample_id in range(num_samples):
        elo_row = {}
        base_quality = np.random.uniform(0.4, 0.9)  # True quality of sample
        
        for annotator in annotators:
            # Each annotator's Elo reflects the true quality with some noise
            noise = np.random.normal(0, 100)  # Annotator-specific noise
            elo_score = 1500 + (base_quality - 0.65) * 500 + noise
            elo_row[annotator] = elo_score
        
        elo_data[sample_id] = elo_row
    
    elo_df = pd.DataFrame.from_dict(elo_data, orient='index')
    elo_df.to_csv(os.path.join(rankings_dir, "elo_values.csv"))
    
    # Create mock metric files that correlate with Elo
    metrics = ['bleu', 'meteor', 'rouge', 'bleurt']
    
    for metric in metrics:
        metric_data = {}
        
        for sample_id in range(num_samples):
            metric_row = {}
            
            for annotator in annotators:
                elo_score = elo_df.loc[sample_id, annotator]
                # Convert Elo to metric score with correlation
                if metric == 'bleu':
                    base_score = (elo_score - 1500) / 500 * 0.3 + 0.6  # Scale to 0.3-0.9
                    noise = np.random.normal(0, 0.1)
                elif metric == 'meteor':
                    base_score = (elo_score - 1500) / 500 * 0.25 + 0.7  # Scale to 0.45-0.95  
                    noise = np.random.normal(0, 0.08)
                elif metric == 'rouge':
                    base_score = (elo_score - 1500) / 500 * 0.35 + 0.55  # Scale to 0.2-0.9
                    noise = np.random.normal(0, 0.12)
                else:  # bleurt
                    base_score = (elo_score - 1500) / 500 * 1.5  # Scale to -1.5 to 1.5
                    noise = np.random.normal(0, 0.3)
                
                final_score = base_score + noise
                
                # Clip to reasonable ranges
                if metric != 'bleurt':
                    final_score = np.clip(final_score, 0, 1)
                else:
                    final_score = np.clip(final_score, -2, 2)
                
                metric_row[annotator] = final_score
            
            metric_data[sample_id] = metric_row
        
        metric_df = pd.DataFrame.from_dict(metric_data, orient='index')
        metric_df.to_csv(os.path.join(rankings_dir, f"{metric}_values.csv"))
    
    # Create winner annotations file
    winner_data = []
    for sample_id in range(num_samples):
        winner_data.append({
            'sample_id': sample_id,
            'text': f"Sample {sample_id} annotation text",
            'winner_annotation': json.dumps([{"text": f"Winner annotation for sample {sample_id}"}])
        })
    
    winner_df = pd.DataFrame(winner_data)
    winner_df.to_csv(os.path.join(data_dir, "winner_annotations.csv"), index=False)
    
    print(f"✅ Created mock causal data in {test_dir}")
    print(f"   Samples: {num_samples}")  
    print(f"   Annotators: {num_annotators}")
    print(f"   Metrics: {metrics}")
    
    return test_dir

def test_pairwise_method(test_dir):
    """Test the pairwise logistic regression method"""
    print("\n🧪 Testing pairwise logistic regression...")
    
    # Modify config temporarily to point to test directory
    import config
    original_datasets_dir = config.DATASETS_DIR
    config.DATASETS_DIR = os.path.dirname(test_dir)
    
    try:
        # Import and test pairwise regression
        sys.path.insert(0, os.path.join(project_root, 'core_scripts'))
        from pairwise_logistic_regression import main as pairwise_main
        from linear_optimization import load_data
        
        # Test data loading
        dataset_name = os.path.basename(test_dir)
        elo_df, metric_dfs = load_data(dataset_name)
        
        if elo_df is None or not metric_dfs:
            print("❌ Failed to load test data")
            return False
        
        print(f"✅ Loaded test data: {len(elo_df)} samples, {len(metric_dfs)} metrics")
        
        # Run pairwise optimization
        from pairwise_logistic_regression import (
            prepare_pairwise_data, cross_validate_pairwise_model,
            evaluate_pairwise_preference_accuracy, save_pairwise_results
        )
        
        X_delta, y, sample_info, metrics = prepare_pairwise_data(elo_df, metric_dfs, dataset_name)
        
        if len(X_delta) == 0:
            print("❌ No pairwise data prepared")
            return False
        
        print(f"✅ Prepared pairwise data: {X_delta.shape}")
        
        # Cross-validate
        weights, cv_scores, _ = cross_validate_pairwise_model(X_delta, y, metrics)
        print(f"✅ Cross-validation complete: accuracy = {np.mean(cv_scores):.4f}")
        
        # Evaluate
        accuracy, predictions_df = evaluate_pairwise_preference_accuracy(
            weights, metric_dfs, elo_df, metrics
        )
        print(f"✅ Evaluation complete: accuracy = {accuracy:.4f}")
        
        # Save results
        results_df = save_pairwise_results(weights, metrics, cv_scores, predictions_df, dataset_name)
        print("✅ Results saved successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in pairwise testing: {e}")
        return False
    finally:
        # Restore original config
        config.DATASETS_DIR = original_datasets_dir

def test_correlation_method(test_dir):
    """Test the correlation-based optimization method"""
    print("\n🧪 Testing correlation-based optimization...")
    
    # Modify config temporarily
    import config
    original_datasets_dir = config.DATASETS_DIR
    config.DATASETS_DIR = os.path.dirname(test_dir)
    
    try:
        sys.path.insert(0, os.path.join(project_root, 'core_scripts'))
        from linear_optimization import (
            load_data, prepare_data_for_optimization, 
            cross_validate_weights, save_results
        )
        
        # Test data loading
        dataset_name = os.path.basename(test_dir)
        elo_df, metric_dfs = load_data(dataset_name)
        
        if elo_df is None or not metric_dfs:
            print("❌ Failed to load test data")
            return False
        
        print(f"✅ Loaded test data: {len(elo_df)} samples, {len(metric_dfs)} metrics")
        
        # Prepare data
        data = prepare_data_for_optimization(elo_df, metric_dfs, dataset_name)
        
        if data is None or len(data) == 0:
            print("❌ No data prepared")
            return False
        
        print(f"✅ Prepared data: {data.shape}")
        
        # Get metrics and optimize
        metrics = [col for col in data.columns if col != 'elo']
        
        optimal_weights, correlations = cross_validate_weights(data, metrics)
        print(f"✅ Cross-validation complete: correlation = {np.mean(correlations):.4f}")
        
        # Save results
        results_df = save_results(optimal_weights, metrics, correlations, dataset_name, data)
        print("✅ Results saved successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in correlation testing: {e}")
        return False
    finally:
        # Restore original config
        config.DATASETS_DIR = original_datasets_dir

def main():
    """Main test function"""
    print("🚀 Testing Dual-Method Optimization Pipeline")
    print("=" * 60)
    
    # Test 1: Pairwise logistic regression (summary data)
    print("\n📊 TEST 1: Pairwise Logistic Regression")
    print("-" * 40)
    
    summary_dir = create_mock_summary_data(num_samples=100)
    pairwise_success = test_pairwise_method(summary_dir)
    
    # Clean up
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)
    
    # Test 2: Correlation-based optimization (causal data)
    print("\n📊 TEST 2: Correlation-based Optimization")
    print("-" * 40)
    
    causal_dir = create_mock_causal_data(num_samples=50, num_annotators=10)
    correlation_success = test_correlation_method(causal_dir)
    
    # Clean up
    if os.path.exists(causal_dir):
        shutil.rmtree(causal_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if pairwise_success:
        print("✅ Pairwise Logistic Regression: PASSED")
    else:
        print("❌ Pairwise Logistic Regression: FAILED")
    
    if correlation_success:
        print("✅ Correlation-based Optimization: PASSED")
    else:
        print("❌ Correlation-based Optimization: FAILED")
    
    if pairwise_success and correlation_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The dual-method optimization pipeline is working correctly.")
        print("\nUsage:")
        print("- For summarize_feedback/hh_rlhf: Uses pairwise logistic regression")
        print("- For causal_relations: Uses correlation-based optimization")
        print("- Run: python run_pipeline.py --dataset <dataset_name>")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)