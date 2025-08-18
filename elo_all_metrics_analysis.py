#!/usr/bin/env python3
"""
Elo-based Correlation Optimization for Summarization Dataset
Using ALL NLP metrics: BLEU, ROUGE-L, METEOR, BLEURT, Verbatim

Treats each model/human as a "player" in a chess tournament where preference 
comparisons are "games". Calculates Elo rankings and optimizes metric weights 
to maximize correlation with Elo scores.
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import os
import glob
import sys
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# Import metric calculation functions from existing codebase
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core_scripts'))

# Import metrics calculation from existing code
import tensorflow as tf
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bleurt import score
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class EloRankingSystem:
    """Implements Elo ranking system for model/human comparison"""
    
    def __init__(self, k_factor=32, initial_rating=1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = defaultdict(lambda: initial_rating)
        self.games_played = defaultdict(int)
        
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for player A vs player B"""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, player_a, player_b, actual_score_a):
        """
        Update Elo ratings after a game
        actual_score_a: 1 if A wins, 0 if B wins, 0.5 for draw
        """
        rating_a = self.ratings[player_a]
        rating_b = self.ratings[player_b]
        
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        # Update ratings
        new_rating_a = rating_a + self.k_factor * (actual_score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - actual_score_a) - expected_b)
        
        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b
        
        self.games_played[player_a] += 1
        self.games_played[player_b] += 1
    
    def get_rankings(self):
        """Get current rankings sorted by Elo rating"""
        return dict(sorted(self.ratings.items(), key=lambda x: x[1], reverse=True))

class AllMetricsEloAnalyzer:
    """Main class for Elo-based analysis using ALL NLP metrics"""
    
    def __init__(self, data_dir="datasets/summarize_feedback"):
        self.data_dir = data_dir
        self.elo_system = EloRankingSystem()
        
        # Initialize all metric scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Initialize BLEURT with CPU fallback (skip for faster testing)
        print("⚠️ Skipping BLEURT initialization for faster testing")
        self.bleurt_scorer = None
            
        self.comparisons = []
        self.summary_data = []
        
    def calculate_verbatim_score(self, reference, candidate):
        """Calculate verbatim matching score"""
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        if len(ref_words) == 0:
            return 0.0
        return len(ref_words.intersection(cand_words)) / len(ref_words)
    
    def calculate_all_metrics(self, reference, candidate):
        """Calculate all NLP metrics for a single summary"""
        metrics = {}
        
        try:
            # BLEU score
            ref_tokens = reference.split()
            cand_tokens = candidate.split()
            if len(cand_tokens) > 0:
                metrics['bleu'] = sentence_bleu([ref_tokens], cand_tokens)
            else:
                metrics['bleu'] = 0.0
                
            # ROUGE-L score (just one type of ROUGE)
            rouge_scores = self.rouge_scorer.score(reference, candidate)
            metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
            
            # METEOR score
            metrics['meteor'] = meteor_score([reference], candidate)
            
            # Verbatim score
            metrics['verbatim'] = self.calculate_verbatim_score(reference, candidate)
            
            # BLEURT score (if available)
            if self.bleurt_scorer is not None:
                try:
                    bleurt_scores = self.bleurt_scorer.score(
                        references=[reference],
                        candidates=[candidate]
                    )
                    metrics['bleurt'] = bleurt_scores[0]
                except Exception as e:
                    print(f"BLEURT error: {e}")
                    metrics['bleurt'] = 0.0
            else:
                metrics['bleurt'] = 0.0
                
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return zero scores if calculation fails
            metrics = {
                'bleu': 0.0,
                'rouge_l': 0.0,
                'meteor': 0.0,
                'verbatim': 0.0,
                'bleurt': 0.0
            }
        
        return metrics
        
    def load_and_process_data(self, max_files=None):
        """Load comparison data and extract player information"""
        print("Loading comparison data...")
        
        comparison_files = glob.glob(f"{self.data_dir}/comparisons/*.json")
        if max_files:
            comparison_files = comparison_files[:max_files]
            
        for file_path in comparison_files:
            print(f"Processing {os.path.basename(file_path)}...")
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        
                        # Extract players and winner
                        summaries = data['summaries']
                        if len(summaries) != 2:
                            continue
                            
                        player_a = summaries[0]['policy']
                        player_b = summaries[1]['policy']
                        winner_idx = data['choice']  # 0 or 1
                        
                        # Store comparison for Elo calculation
                        comparison = {
                            'player_a': player_a,
                            'player_b': player_b,
                            'winner': player_a if winner_idx == 0 else player_b,
                            'original_post': data['info'].get('post', data['info'].get('article', '')),
                            'summary_a': summaries[0]['text'],
                            'summary_b': summaries[1]['text']
                        }
                        self.comparisons.append(comparison)
                        
                        # Store individual summaries for metric calculation
                        for i, summary in enumerate(summaries):
                            summary_info = {
                                'player': summary['policy'],
                                'text': summary['text'],
                                'original_post': comparison['original_post'],
                                'comparison_id': len(self.comparisons) - 1,
                                'is_winner': (i == winner_idx)
                            }
                            self.summary_data.append(summary_info)
                            
                    except Exception as e:
                        print(f"Error processing line {line_num} in {file_path}: {e}")
                        continue
        
        print(f"Loaded {len(self.comparisons)} comparisons with {len(self.summary_data)} total summaries")
        
        # Show player statistics
        player_counts = Counter([comp['player_a'] for comp in self.comparisons] + 
                               [comp['player_b'] for comp in self.comparisons])
        print("\nPlayer participation:")
        for player, count in player_counts.most_common():
            print(f"  {player}: {count} games")
            
    def calculate_elo_rankings(self):
        """Calculate Elo rankings from all comparisons"""
        print("\nCalculating Elo rankings...")
        
        for comp in self.comparisons:
            # Determine actual score (1 if A wins, 0 if B wins)
            actual_score_a = 1.0 if comp['winner'] == comp['player_a'] else 0.0
            
            # Update Elo ratings
            self.elo_system.update_ratings(
                comp['player_a'], 
                comp['player_b'], 
                actual_score_a
            )
        
        # Get final rankings
        rankings = self.elo_system.get_rankings()
        
        print("\nFinal Elo Rankings:")
        for i, (player, rating) in enumerate(rankings.items(), 1):
            games = self.elo_system.games_played[player]
            print(f"  {i:2d}. {player:25s}: {rating:7.1f} ({games} games)")
            
        return rankings
    
    def calculate_all_metrics_batch(self):
        """Calculate all NLP metrics for all summaries"""
        print("\nCalculating ALL NLP metrics (BLEU, ROUGE-L, METEOR, Verbatim)...")
        print("(Skipping BLEURT for faster testing)")
        
        metrics_data = []
        batch_size = 500  # Larger batches since no BLEURT
        
        for i in range(0, len(self.summary_data), batch_size):
            batch = self.summary_data[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(self.summary_data)-1)//batch_size + 1}...")
            
            # Prepare batch data for BLEURT
            if self.bleurt_scorer is not None:
                references = [s['original_post'] for s in batch]
                candidates = [s['text'] for s in batch]
                
                try:
                    bleurt_scores = self.bleurt_scorer.score(references, candidates)
                except Exception as e:
                    print(f"BLEURT batch error: {e}")
                    bleurt_scores = [0.0] * len(batch)
            else:
                bleurt_scores = [0.0] * len(batch)
            
            # Calculate other metrics for each summary in batch
            for j, summary in enumerate(batch):
                try:
                    # Calculate individual metrics
                    metrics = self.calculate_all_metrics(summary['original_post'], summary['text'])
                    
                    # Override BLEURT with batch result
                    metrics['bleurt'] = bleurt_scores[j]
                    
                    metric_row = {
                        'player': summary['player'],
                        'comparison_id': summary['comparison_id'],
                        'is_winner': summary['is_winner'],
                        **metrics  # Add all metric scores
                    }
                    metrics_data.append(metric_row)
                    
                except Exception as e:
                    print(f"Error calculating metrics for summary: {e}")
                    continue
            
            # Clear memory after each batch
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        print(f"Calculated all metrics for {len(metrics_data)} summaries")
        return pd.DataFrame(metrics_data)
    
    def aggregate_player_metrics(self, metrics_df, elo_rankings):
        """Aggregate metrics by player and align with Elo rankings"""
        print("\nAggregating metrics by player...")
        
        # Calculate mean metrics for each player
        metric_cols = ['bleu', 'rouge_l', 'meteor', 'verbatim']
        player_metrics = metrics_df.groupby('player')[metric_cols].mean().reset_index()
        
        # Add Elo ratings
        player_metrics['elo_rating'] = player_metrics['player'].map(elo_rankings)
        
        # Sort by Elo rating
        player_metrics = player_metrics.sort_values('elo_rating', ascending=False)
        
        print(f"Aggregated metrics for {len(player_metrics)} players")
        print("\nPlayer metrics summary:")
        for _, row in player_metrics.head(10).iterrows():
            print(f"  {row['player'][:20]:20s}: Elo={row['elo_rating']:7.1f}, "
                  f"BLEU={row['bleu']:.3f}, ROUGE-L={row['rouge_l']:.3f}, "
                  f"METEOR={row['meteor']:.3f}, Verbatim={row['verbatim']:.3f}")
        
        return player_metrics
    
    def optimize_metric_weights(self, player_metrics):
        """Find optimal linear combination of ALL metrics to maximize correlation with Elo"""
        print("\nOptimizing weights for linear combination of ALL metrics...")
        
        # Define metric columns (excluding BLEURT for faster testing)
        metric_cols = ['bleu', 'rouge_l', 'meteor', 'verbatim']
        
        # Extract metric matrix and Elo ratings
        X = player_metrics[metric_cols].values
        y = player_metrics['elo_rating'].values
        
        def objective(weights):
            """Negative Spearman correlation (to minimize)"""
            combined_score = X @ weights
            correlation, _ = spearmanr(combined_score, y)
            return -correlation if not np.isnan(correlation) else 0
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: all weights between 0 and 1
        bounds = [(0, 1) for _ in metric_cols]
        
        # Initial guess: equal weights
        initial_weights = np.ones(len(metric_cols)) / len(metric_cols)
        
        # Cross-validation
        kf = KFold(n_splits=min(5, len(player_metrics)), shuffle=True, random_state=42)
        cv_correlations = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            if len(val_idx) < 3:  # Skip if validation set too small
                continue
                
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Optimize on training set
            def fold_objective(weights):
                combined_score = X_train @ weights
                correlation, _ = spearmanr(combined_score, y_train)
                return -correlation if not np.isnan(correlation) else 0
            
            result = minimize(
                fold_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # Evaluate on validation set
            if result.success:
                val_combined = X_val @ result.x
                val_corr, _ = spearmanr(val_combined, y_val)
                cv_correlations.append(val_corr)
                print(f"  Fold {fold+1}: correlation = {val_corr:.4f}")
        
        # Final optimization on full dataset
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            final_combined = X @ optimal_weights
            final_correlation, _ = spearmanr(final_combined, y)
            
            print(f"\nOptimization Results:")
            print(f"  Final correlation: {final_correlation:.4f}")
            if cv_correlations:
                print(f"  CV mean correlation: {np.mean(cv_correlations):.4f} ± {np.std(cv_correlations):.4f}")
            print(f"  Optimal weights:")
            for metric, weight in zip(metric_cols, optimal_weights):
                print(f"    {metric:10s}: {weight:.4f}")
                
            return optimal_weights, metric_cols, final_correlation, cv_correlations
        else:
            print("Optimization failed!")
            return None, None, None, None
    
    def create_visualizations(self, player_metrics, optimal_weights, metric_cols, correlation, cv_correlations):
        """Create comprehensive visualization plots"""
        print("\nCreating visualizations...")
        
        # Create output directory
        os.makedirs("datasets/summarize_feedback/elo_all_metrics", exist_ok=True)
        
        # Calculate combined scores
        X = player_metrics[metric_cols].values
        combined_scores = X @ optimal_weights
        player_metrics['combined_score'] = combined_scores
        
        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Elo-based Optimization: ALL NLP Metrics (BLEU, ROUGE-L, METEOR, Verbatim)', fontsize=14)
        
        # Plot 1: Elo vs Combined Score correlation
        axes[0,0].scatter(player_metrics['elo_rating'], player_metrics['combined_score'], 
                         alpha=0.8, s=100, c='blue')
        axes[0,0].set_xlabel('Elo Rating')
        axes[0,0].set_ylabel('Combined Metric Score')
        axes[0,0].set_title(f'Elo vs Combined Score\n(Spearman ρ = {correlation:.4f})')
        
        # Add best fit line
        z = np.polyfit(player_metrics['elo_rating'], player_metrics['combined_score'], 1)
        p = np.poly1d(z)
        axes[0,0].plot(player_metrics['elo_rating'], p(player_metrics['elo_rating']), "r--", alpha=0.8)
        
        # Add player names (shorter labels)
        for _, row in player_metrics.iterrows():
            label = row['player']
            if label == 'ref':
                label = 'HUMAN'
            elif 'sup4' in label:
                label = label.replace('sup4_ppo_rm3_kl', 'kl')
            axes[0,0].annotate(label[:8], 
                             (row['elo_rating'], row['combined_score']), 
                             fontsize=9, alpha=0.8, ha='center')
        
        # Plot 2: Optimal weights bar chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[0,1].bar(range(len(metric_cols)), optimal_weights, color=colors)
        axes[0,1].set_xticks(range(len(metric_cols)))
        axes[0,1].set_xticklabels([m.upper() for m in metric_cols])
        axes[0,1].set_ylabel('Weight')
        axes[0,1].set_title('Optimal Metric Weights')
        
        # Add weight values on bars
        for bar, weight in zip(bars, optimal_weights):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{weight:.3f}', ha='center', va='bottom')
        
        # Plot 3: Individual metric correlations with Elo
        individual_corrs = []
        for metric in metric_cols:
            corr, _ = spearmanr(player_metrics[metric], player_metrics['elo_rating'])
            individual_corrs.append(corr)
        
        bars = axes[1,0].bar(range(len(metric_cols)), individual_corrs, color=colors)
        axes[1,0].set_xticks(range(len(metric_cols)))
        axes[1,0].set_xticklabels([m.upper() for m in metric_cols])
        axes[1,0].set_ylabel('Spearman Correlation with Elo')
        axes[1,0].set_title('Individual Metric Correlations')
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, individual_corrs):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, 
                          bar.get_height() + (0.02 if corr >= 0 else -0.05),
                          f'{corr:.3f}', ha='center', va='bottom' if corr >= 0 else 'top')
        
        # Plot 4: Cross-validation results or weight distribution pie chart
        if cv_correlations and len(cv_correlations) > 0:
            axes[1,1].boxplot(cv_correlations)
            axes[1,1].set_ylabel('Correlation')
            axes[1,1].set_title(f'Cross-Validation Results\n(Mean: {np.mean(cv_correlations):.3f})')
            axes[1,1].set_xticklabels(['5-Fold CV'])
        else:
            # Pie chart of weight distribution
            axes[1,1].pie(optimal_weights, labels=[m.upper() for m in metric_cols], 
                         autopct='%1.1f%%', colors=colors)
            axes[1,1].set_title('Weight Distribution')
        
        plt.tight_layout()
        plt.savefig('datasets/summarize_feedback/elo_all_metrics/elo_all_metrics_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        results_file = 'datasets/summarize_feedback/elo_all_metrics/elo_all_metrics_analysis.csv'
        player_metrics.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
        
        # Save weights
        weights_df = pd.DataFrame({
            'metric': metric_cols,
            'weight': optimal_weights,
            'individual_correlation': individual_corrs
        })
        weights_file = 'datasets/summarize_feedback/elo_all_metrics/optimal_weights_all_metrics.csv'
        weights_df.to_csv(weights_file, index=False)
        print(f"Weights saved to {weights_file}")
    
    def run_full_analysis(self, max_files=3):
        """Run the complete Elo-based analysis pipeline with ALL metrics"""
        print("=" * 70)
        print("ELO-BASED OPTIMIZATION: ALL NLP METRICS")
        print("BLEU + ROUGE-L + METEOR + Verbatim")
        print("=" * 70)
        
        # Step 1: Load data
        self.load_and_process_data(max_files=max_files)
        
        # Step 2: Calculate Elo rankings
        elo_rankings = self.calculate_elo_rankings()
        
        # Step 3: Calculate ALL metrics
        metrics_df = self.calculate_all_metrics_batch()
        
        # Step 4: Aggregate by player
        player_metrics = self.aggregate_player_metrics(metrics_df, elo_rankings)
        
        # Step 5: Optimize metric weights
        optimal_weights, metric_cols, correlation, cv_correlations = self.optimize_metric_weights(player_metrics)
        
        # Step 6: Create visualizations
        if optimal_weights is not None:
            self.create_visualizations(player_metrics, optimal_weights, metric_cols, correlation, cv_correlations)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        
        return {
            'elo_rankings': elo_rankings,
            'player_metrics': player_metrics,
            'optimal_weights': optimal_weights,
            'metric_cols': metric_cols,
            'correlation': correlation,
            'cv_correlations': cv_correlations
        }

if __name__ == "__main__":
    # Run the analysis with all metrics
    analyzer = AllMetricsEloAnalyzer()
    results = analyzer.run_full_analysis(max_files=3)  # Start with 3 files for testing