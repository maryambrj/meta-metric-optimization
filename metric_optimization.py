#!/usr/bin/env python3
"""
Metric Optimization for Summarization Dataset

Finds optimal linear combination weights for NLP metrics to maximize 
Spearman correlation with Elo rankings.

Process:
1. Load metric scores from metric_calculator.py output
2. Load Elo rankings from elo_tournament.py output  
3. Calculate participant rankings based on metric combinations
4. Optimize weights to maximize Spearman correlation with Elo rankings
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class MetricOptimizer:
    """Optimize metric weights for maximum correlation with Elo rankings"""
    
    def __init__(self, output_file=None, resume=True):
        self.metrics = ['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL', 'verbatim', 'bleurt']
        self.metric_data = None
        self.elo_rankings = None
        self.participant_scores = None
        self.output_file = output_file
        self.weights_file = None
        self.summary_file = None
        self.iteration_count = 0
        self.resume = resume
        
        # Initialize output files if provided
        if self.output_file:
            self.weights_file = self.output_file.replace('.csv', '_weights.csv')
            self.summary_file = self.output_file.replace('.csv', '_summary.txt')
            self._initialize_output_files()
    
    def _initialize_output_files(self):
        """Initialize output files with headers or check for existing files"""
        try:
            # Check for existing files
            results_exists = os.path.exists(self.output_file)
            weights_exists = os.path.exists(self.weights_file)
            summary_exists = os.path.exists(self.summary_file)
            
            if self.resume and (results_exists or weights_exists or summary_exists):
                print(f"📄 Found existing optimization files:")
                if results_exists:
                    with open(self.output_file, 'r') as f:
                        lines = len(f.readlines())
                    print(f"   Results: {self.output_file} ({lines-1} iterations)")
                if weights_exists:
                    with open(self.weights_file, 'r') as f:
                        lines = len(f.readlines())
                    print(f"   Weights: {self.weights_file} ({lines-1} iterations)")
                if summary_exists:
                    print(f"   Summary: {self.summary_file}")
                
                # Get last iteration number
                self.iteration_count = self._get_last_iteration()
                if self.iteration_count > 0:
                    print(f"🔄 Resume mode: Will continue from iteration {self.iteration_count + 1}")
                    return
            elif not self.resume and (results_exists or weights_exists or summary_exists):
                print(f"🗑️ Starting fresh: Removing existing files...")
                if results_exists:
                    os.remove(self.output_file)
                if weights_exists:
                    os.remove(self.weights_file)
                if summary_exists:
                    os.remove(self.summary_file)
            
            # Initialize new files
            with open(self.output_file, 'w') as f:
                f.write("iteration,participant,metric_rank,metric_score,elo_rank,elo_rating,correlation\n")
            
            with open(self.weights_file, 'w') as f:
                header = "iteration," + ",".join(self.metrics) + ",correlation\n"
                f.write(header)
            
            with open(self.summary_file, 'w') as f:
                f.write("Metric Optimization Results - Incremental\n")
                f.write("=" * 40 + "\n\n")
            
            print(f"📁 Initialized new output files:")
            print(f"   Results: {self.output_file}")
            print(f"   Weights: {self.weights_file}")
            print(f"   Summary: {self.summary_file}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize output files: {e}")
    
    def _get_last_iteration(self):
        """Get the last iteration number from existing files"""
        try:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 1:  # Skip header
                    last_line = lines[-1].strip()
                    if last_line:
                        return int(last_line.split(',')[0])
        except Exception as e:
            print(f"⚠️ Warning: Could not read last iteration: {e}")
        
        return 0
    
    def _write_iteration_results(self, weights, correlation, results_df):
        """Write results for current iteration"""
        try:
            self.iteration_count += 1
            
            # Write weights
            with open(self.weights_file, 'a') as f:
                weight_str = ",".join(f"{w:.6f}" for w in weights)
                f.write(f"{self.iteration_count},{weight_str},{correlation:.6f}\n")
            
            # Write detailed results
            with open(self.output_file, 'a') as f:
                for _, row in results_df.iterrows():
                    f.write(f"{self.iteration_count},{row['participant']},{row['metric_rank']},{row['metric_score']:.6f},{row['rank']},{row['final_rating']:.2f},{correlation:.6f}\n")
            
            # Update summary
            with open(self.summary_file, 'a') as f:
                f.write(f"\nIteration {self.iteration_count}:\n")
                f.write(f"  Correlation: {correlation:.6f}\n")
                f.write(f"  Weights: {dict(zip(self.metrics, weights))}\n")
            
            print(f"💾 Saved iteration {self.iteration_count} results (correlation: {correlation:.4f})")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not write iteration results: {e}")
    
    def load_data(self, metric_file, elo_file):
        """Load metric scores and Elo rankings"""
        print("📊 Loading data...")
        
        # Load metric scores
        if not os.path.exists(metric_file):
            print(f"❌ Metric file not found: {metric_file}")
            return False
        
        self.metric_data = pd.read_csv(metric_file)
        print(f"✅ Loaded {len(self.metric_data):,} metric samples")
        
        # Load Elo rankings
        if not os.path.exists(elo_file):
            print(f"❌ Elo file not found: {elo_file}")
            return False
        
        self.elo_rankings = pd.read_csv(elo_file)
        print(f"✅ Loaded {len(self.elo_rankings)} Elo rankings")
        
        return True
    
    def calculate_participant_scores(self, weights):
        """
        Calculate average metric scores for each participant using given weights
        
        Args:
            weights (array): Weights for metrics [bleu, meteor, rouge1, rouge2, rougeL, verbatim, bleurt]
        
        Returns:
            dict: participant -> average weighted score
        """
        participant_scores = {}
        
        # Process each sample
        for _, row in self.metric_data.iterrows():
            # Get policies (participants)
            policy1 = row['summary1_policy']
            policy2 = row['summary2_policy']
            
            # Calculate weighted scores for both summaries
            score1 = 0.0
            score2 = 0.0
            
            for i, metric in enumerate(self.metrics):
                col1 = f'summary1_{metric}'
                col2 = f'summary2_{metric}'
                
                if col1 in row and col2 in row:
                    score1 += weights[i] * row[col1]
                    score2 += weights[i] * row[col2]
            
            # Add to participant scores
            if policy1 not in participant_scores:
                participant_scores[policy1] = []
            if policy2 not in participant_scores:
                participant_scores[policy2] = []
            
            participant_scores[policy1].append(score1)
            participant_scores[policy2].append(score2)
        
        # Calculate average scores
        avg_scores = {}
        for participant, scores in participant_scores.items():
            avg_scores[participant] = np.mean(scores)
        
        return avg_scores
    
    def calculate_metric_ranking(self, weights):
        """Calculate participant ranking based on weighted metrics"""
        # Get average scores for each participant
        participant_scores = self.calculate_participant_scores(weights)
        
        # Create ranking DataFrame
        ranking_data = []
        for participant, score in participant_scores.items():
            ranking_data.append({
                'participant': participant,
                'metric_score': score
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('metric_score', ascending=False)
        ranking_df['metric_rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df
    
    def objective_function(self, weights):
        """Objective function: negative Spearman correlation"""
        try:
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate metric-based ranking
            metric_ranking = self.calculate_metric_ranking(weights)
            
            # Merge with Elo rankings
            merged = pd.merge(
                metric_ranking[['participant', 'metric_rank', 'metric_score']], 
                self.elo_rankings[['player', 'rank', 'final_rating']], 
                left_on='participant', 
                right_on='player', 
                how='inner'
            )
            
            if len(merged) < 2:
                return 1.0  # Return worst correlation if insufficient data
            
            # Calculate Spearman correlation
            correlation, _ = spearmanr(merged['metric_rank'], merged['rank'])
            
            # Save results incrementally if correlation is valid and output file is set
            if not np.isnan(correlation) and self.output_file:
                self._write_iteration_results(weights, correlation, merged)
            
            # Return negative correlation (minimize negative = maximize positive)
            return -correlation if not np.isnan(correlation) else 1.0
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1.0
    
    def optimize_weights(self):
        """Find optimal weights using scipy optimization"""
        print("🔧 Optimizing metric weights...")
        
        # Initial weights (equal)
        initial_weights = np.ones(len(self.metrics)) / len(self.metrics)
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in self.metrics]
        
        # Optimize
        result = minimize(
            self.objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'disp': True}
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # Normalize
            correlation = -result.fun
            
            print(f"✅ Optimization successful!")
            print(f"   Final correlation: {correlation:.4f}")
            
            return optimal_weights, correlation
        else:
            print(f"⚠️ Optimization failed: {result.message}")
            return initial_weights, 0.0
    
    def cross_validate_weights(self, n_folds=5):
        """Cross-validate the optimization with different data splits"""
        print(f"🔄 Cross-validating with {n_folds} folds...")
        
        # Add fold info to summary
        if self.summary_file:
            with open(self.summary_file, 'a') as f:
                f.write(f"\n=== Cross-Validation ({n_folds} folds) ===\n")
        
        # For simplicity, just run optimization multiple times with random subsets
        correlations = []
        weight_sets = []
        
        for fold in range(n_folds):
            print(f"\n   Fold {fold + 1}/{n_folds}")
            
            # Add fold marker to summary
            if self.summary_file:
                with open(self.summary_file, 'a') as f:
                    f.write(f"\nFold {fold + 1}:\n")
            
            # Use a random subset for this fold
            subset_size = int(0.8 * len(self.metric_data))
            subset_indices = np.random.choice(len(self.metric_data), subset_size, replace=False)
            original_data = self.metric_data.copy()
            self.metric_data = self.metric_data.iloc[subset_indices].reset_index(drop=True)
            
            # Optimize weights
            weights, correlation = self.optimize_weights()
            correlations.append(correlation)
            weight_sets.append(weights)
            
            # Restore original data
            self.metric_data = original_data
        
        # Calculate average weights and correlation
        avg_weights = np.mean(weight_sets, axis=0)
        avg_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
        
        print(f"\n📊 Cross-validation results:")
        print(f"   Mean correlation: {avg_correlation:.4f} ± {std_correlation:.4f}")
        print(f"   Individual correlations: {[f'{c:.4f}' for c in correlations]}")
        
        # Save cross-validation summary
        if self.summary_file:
            with open(self.summary_file, 'a') as f:
                f.write(f"\nCross-validation Summary:\n")
                f.write(f"  Mean correlation: {avg_correlation:.4f} ± {std_correlation:.4f}\n")
                f.write(f"  Individual correlations: {correlations}\n")
                f.write(f"  Average weights: {dict(zip(self.metrics, avg_weights))}\n")
        
        return avg_weights, avg_correlation, std_correlation
    
    def analyze_sample_distribution(self):
        """Analyze sample distribution across participants"""
        print("\n📊 Sample Distribution Analysis:")
        
        participant_counts = {}
        
        # Count samples per participant
        for _, row in self.metric_data.iterrows():
            policy1 = row['summary1_policy']
            policy2 = row['summary2_policy']
            
            participant_counts[policy1] = participant_counts.get(policy1, 0) + 1
            participant_counts[policy2] = participant_counts.get(policy2, 0) + 1
        
        # Create distribution DataFrame
        dist_df = pd.DataFrame(list(participant_counts.items()), columns=['participant', 'sample_count'])
        dist_df = dist_df.sort_values('sample_count', ascending=False)
        
        print(f"   Total participants: {len(dist_df)}")
        print(f"   Min samples: {dist_df['sample_count'].min()}")
        print(f"   Max samples: {dist_df['sample_count'].max()}")
        print(f"   Mean samples: {dist_df['sample_count'].mean():.1f}")
        print(f"   Median samples: {dist_df['sample_count'].median():.1f}")
        
        print(f"\n   Top 10 most active participants:")
        for _, row in dist_df.head(10).iterrows():
            print(f"     {row['participant']:<25}: {row['sample_count']:,} samples")
        
        print(f"\n   Bottom 10 least active participants:")
        for _, row in dist_df.tail(10).iterrows():
            print(f"     {row['participant']:<25}: {row['sample_count']:,} samples")
        
        return dist_df
    
    def calculate_participant_scores_filtered(self, weights, min_samples=50):
        """
        Calculate participant scores but only include those with enough samples
        
        Args:
            weights (array): Weights for metrics
            min_samples (int): Minimum number of samples required
        """
        # First pass: count samples per participant
        participant_counts = {}
        for _, row in self.metric_data.iterrows():
            policy1 = row['summary1_policy']
            policy2 = row['summary2_policy']
            participant_counts[policy1] = participant_counts.get(policy1, 0) + 1
            participant_counts[policy2] = participant_counts.get(policy2, 0) + 1
        
        # Filter participants with enough samples
        qualified_participants = {p for p, count in participant_counts.items() if count >= min_samples}
        print(f"   Participants with ≥{min_samples} samples: {len(qualified_participants)}/{len(participant_counts)}")
        
        # Second pass: calculate scores only for qualified participants
        participant_scores = {}
        
        for _, row in self.metric_data.iterrows():
            policy1 = row['summary1_policy']
            policy2 = row['summary2_policy']
            
            # Only process if participant is qualified
            if policy1 in qualified_participants or policy2 in qualified_participants:
                # Calculate weighted scores
                score1 = sum(weights[i] * row[f'summary1_{metric}'] 
                           for i, metric in enumerate(self.metrics) 
                           if f'summary1_{metric}' in row)
                score2 = sum(weights[i] * row[f'summary2_{metric}'] 
                           for i, metric in enumerate(self.metrics) 
                           if f'summary2_{metric}' in row)
                
                # Add to qualified participants only
                if policy1 in qualified_participants:
                    if policy1 not in participant_scores:
                        participant_scores[policy1] = []
                    participant_scores[policy1].append(score1)
                
                if policy2 in qualified_participants:
                    if policy2 not in participant_scores:
                        participant_scores[policy2] = []
                    participant_scores[policy2].append(score2)
        
        # Calculate averages
        avg_scores = {}
        for participant, scores in participant_scores.items():
            avg_scores[participant] = np.mean(scores)
        
        return avg_scores, qualified_participants
    
    def analyze_results(self, weights, correlation):
        """Analyze and display optimization results"""
        print(f"\n🎯 METRIC OPTIMIZATION RESULTS")
        print("=" * 50)
        
        # Analyze sample distribution
        dist_df = self.analyze_sample_distribution()
        
        print(f"\n📈 Optimal Weights:")
        for i, metric in enumerate(self.metrics):
            print(f"   {metric.upper():<8}: {weights[i]:.4f}")
        
        print(f"\n📊 Performance:")
        print(f"   Spearman Correlation: {correlation:.4f}")
        
        # Calculate rankings with different minimum sample thresholds
        thresholds = [10, 50, 100]
        
        for min_samples in thresholds:
            print(f"\n🔍 Analysis with min {min_samples} samples per participant:")
            
            # Get filtered scores
            participant_scores, qualified = self.calculate_participant_scores_filtered(weights, min_samples)
            
            if len(participant_scores) < 2:
                print(f"   ⚠️ Not enough qualified participants ({len(participant_scores)})")
                continue
            
            # Create ranking
            ranking_data = [{'participant': p, 'metric_score': s} for p, s in participant_scores.items()]
            metric_ranking = pd.DataFrame(ranking_data)
            metric_ranking = metric_ranking.sort_values('metric_score', ascending=False)
            metric_ranking['metric_rank'] = range(1, len(metric_ranking) + 1)
            
            # Merge with Elo rankings
            merged = pd.merge(
                metric_ranking[['participant', 'metric_rank', 'metric_score']], 
                self.elo_rankings[['player', 'rank', 'final_rating']], 
                left_on='participant', 
                right_on='player', 
                how='inner'
            )
            
            if len(merged) >= 2:
                corr, _ = spearmanr(merged['metric_rank'], merged['rank'])
                print(f"   Spearman correlation: {corr:.4f} ({len(merged)} participants)")
                
                # Show top 5
                print(f"   Top 5 by metrics:")
                for _, row in merged.head(5).iterrows():
                    samples = dist_df[dist_df['participant'] == row['participant']]['sample_count'].iloc[0]
                    print(f"     {row['participant']:<20} (Metric #{row['metric_rank']}, Elo #{row['rank']}, {samples} samples)")
        
        # Return the standard (unfiltered) merged results for compatibility
        metric_ranking = self.calculate_metric_ranking(weights)
        merged = pd.merge(
            metric_ranking[['participant', 'metric_rank', 'metric_score']], 
            self.elo_rankings[['player', 'rank', 'final_rating']], 
            left_on='participant', 
            right_on='player', 
            how='inner'
        )
        
        return merged
    
    def save_results(self, weights, correlation, results_df, output_file):
        """Save optimization results (final summary if incremental writing was used)"""
        if self.output_file:
            # If incremental writing was used, just add final summary
            print(f"\n💾 Results already saved incrementally during optimization")
            print(f"   Total iterations saved: {self.iteration_count}")
            
            # Add final summary to existing summary file
            with open(self.summary_file, 'a') as f:
                f.write(f"\n" + "=" * 40 + "\n")
                f.write(f"FINAL RESULTS\n")
                f.write(f"=" * 40 + "\n")
                f.write(f"Total optimization iterations: {self.iteration_count}\n")
                f.write(f"Final Spearman Correlation: {correlation:.6f}\n\n")
                f.write("Final Optimal Weights:\n")
                for i, metric in enumerate(self.metrics):
                    f.write(f"  {metric}: {weights[i]:.6f}\n")
                f.write(f"\nNumber of participants in final ranking: {len(results_df)}\n")
            
            print(f"💾 Updated final summary in: {self.summary_file}")
        else:
            # Fallback to original behavior if no incremental writing
            weights_df = pd.DataFrame({
                'metric': self.metrics,
                'weight': weights
            })
            
            weights_file = output_file.replace('.csv', '_weights.csv')
            weights_df.to_csv(weights_file, index=False)
            print(f"\n💾 Saved optimal weights to: {weights_file}")
            
            # Save ranking comparison
            results_df.to_csv(output_file, index=False)
            print(f"💾 Saved ranking comparison to: {output_file}")
            
            # Save summary
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Metric Optimization Results\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Spearman Correlation: {correlation:.4f}\n\n")
                f.write("Optimal Weights:\n")
                for i, metric in enumerate(self.metrics):
                    f.write(f"  {metric}: {weights[i]:.4f}\n")
            
            print(f"💾 Saved summary to: {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Optimize metric weights for Elo correlation")
    parser.add_argument("--metric-file", default="metric_scores.csv", help="Metric scores CSV file")
    parser.add_argument("--elo-file", default="datasets/summarize_feedback/elo_tournament_results/elo_tournament_rankings.csv", help="Elo rankings CSV file")
    parser.add_argument("--output", default="metric_optimization_results.csv", help="Output file")
    parser.add_argument("--cross-validate", action="store_true", help="Use cross-validation")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh instead of resuming from existing files")
    
    args = parser.parse_args()
    
    print("🚀 Metric Weight Optimization")
    print("=" * 40)
    
    # Initialize optimizer with output file for incremental writing
    optimizer = MetricOptimizer(output_file=args.output, resume=not args.no_resume)
    
    # Load data
    if not optimizer.load_data(args.metric_file, args.elo_file):
        return
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Optimize weights
    if args.cross_validate:
        weights, correlation, std_corr = optimizer.cross_validate_weights()
        print(f"\n✅ Cross-validated correlation: {correlation:.4f} ± {std_corr:.4f}")
    else:
        weights, correlation = optimizer.optimize_weights()
    
    # Analyze results
    results_df = optimizer.analyze_results(weights, correlation)
    
    # Save results
    optimizer.save_results(weights, correlation, results_df, args.output)
    
    print(f"\n✅ Optimization complete!")


if __name__ == "__main__":
    main()