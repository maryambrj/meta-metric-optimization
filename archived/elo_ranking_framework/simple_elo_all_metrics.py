#!/usr/bin/env python3
"""
Simple Elo-based analysis with all major NLP metrics
Optimized for speed
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
import glob
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from scipy.stats import spearmanr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from gpu_utils import detect_gpu, setup_gpu_for_bleurt, optimize_batch_size_for_gpu, print_gpu_status

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

class SimpleEloAnalyzer:
    def __init__(self, data_dir="datasets/summarize_feedback", enable_bleurt=None, auto_gpu=True):
        self.data_dir = data_dir
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # GPU detection and automatic setup
        if auto_gpu:
            self.gpu_info = print_gpu_status()
        else:
            self.gpu_info = detect_gpu()
        
        # Auto-enable BLEURT if GPU is available (unless explicitly disabled)
        if enable_bleurt is None:
            enable_bleurt = self.gpu_info['tensorflow_gpu']
            if enable_bleurt:
                print("🚀 GPU detected - automatically enabling BLEURT")
            else:
                print("⚡ No TensorFlow GPU - BLEURT disabled for faster processing")
        
        # Initialize BLEURT
        if enable_bleurt:
            try:
                print("Initializing BLEURT with GPU optimization...")
                
                # Setup GPU configuration first
                if self.gpu_info['tensorflow_gpu']:
                    setup_gpu_for_bleurt()
                
                from bleurt import score as bleurt_score
                self.bleurt_scorer = bleurt_score.BleurtScorer(checkpoint="BLEURT-20")
                
                # Set optimal batch size based on GPU memory
                self.bleurt_batch_size = optimize_batch_size_for_gpu(
                    self.gpu_info.get('gpu_memory'), 
                    default_batch_size=32
                )
                print(f"✅ BLEURT initialized with batch size: {self.bleurt_batch_size}")
            except Exception as e:
                print(f"⚠️ BLEURT initialization failed: {e}")
                print("   Make sure you have: pip install bleurt")
                print("   And BLEURT-20 checkpoint downloaded")
                print("Will continue without BLEURT")
                self.bleurt_scorer = None
                self.bleurt_batch_size = 1
        else:
            self.bleurt_scorer = None
            self.bleurt_batch_size = 1
        
    def calculate_metrics(self, reference, candidate):
        """Calculate BLEU, BLEURT, METEOR, ROUGE, Verbatim"""
        metrics = {}
        
        try:
            # BLEU
            ref_tokens = reference.split()
            cand_tokens = candidate.split()
            if len(cand_tokens) > 0:
                metrics['bleu'] = sentence_bleu([ref_tokens], cand_tokens)
            else:
                metrics['bleu'] = 0.0
            
            # ROUGE-L
            rouge_scores = self.rouge_scorer.score(reference, candidate)
            metrics['rouge_l'] = rouge_scores['rougeL'].fmeasure
            
            # METEOR  
            metrics['meteor'] = meteor_score([reference.split()], candidate.split())
            
            # Verbatim
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            if len(ref_words) > 0:
                metrics['verbatim'] = len(ref_words.intersection(cand_words)) / len(ref_words)
            else:
                metrics['verbatim'] = 0.0
            
            # BLEURT (if available)
            if self.bleurt_scorer is not None:
                try:
                    bleurt_scores = self.bleurt_scorer.score(
                        references=[reference],
                        candidates=[candidate]
                    )
                    metrics['bleurt'] = bleurt_scores[0]
                except Exception as e:
                    metrics['bleurt'] = 0.0
            else:
                metrics['bleurt'] = 0.0
                
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {'bleu': 0.0, 'rouge_l': 0.0, 'meteor': 0.0, 'verbatim': 0.0, 'bleurt': 0.0}
        
        return metrics
    
    def load_data(self, max_files=5):
        """Load and process comparison data"""
        print(f"Loading data from {max_files} files...")
        
        comparison_files = glob.glob(f"{self.data_dir}/comparisons/*.json")[:max_files]
        
        comparisons = []
        summary_data = []
        
        for file_path in comparison_files:
            print(f"Processing {os.path.basename(file_path)}...")
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        summaries = data['summaries']
                        if len(summaries) != 2:
                            continue
                            
                        player_a = summaries[0]['policy']
                        player_b = summaries[1]['policy'] 
                        winner_idx = data['choice']
                        
                        comparison = {
                            'player_a': player_a,
                            'player_b': player_b,
                            'winner': player_a if winner_idx == 0 else player_b
                        }
                        comparisons.append(comparison)
                        
                        # Store summaries with original text for metrics
                        original_text = data['info'].get('post', data['info'].get('article', ''))
                        for i, summary in enumerate(summaries):
                            summary_info = {
                                'player': summary['policy'],
                                'text': summary['text'],
                                'original': original_text
                            }
                            summary_data.append(summary_info)
                            
                    except Exception as e:
                        continue
        
        print(f"Loaded {len(comparisons)} comparisons, {len(summary_data)} summaries")
        return comparisons, summary_data
    
    def calculate_elo_rankings(self, comparisons):
        """Calculate Elo rankings"""
        print("Calculating Elo rankings...")
        
        ratings = defaultdict(lambda: 1500)
        
        for comp in comparisons:
            # Update Elo ratings
            rating_a = ratings[comp['player_a']]
            rating_b = ratings[comp['player_b']]
            
            expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
            
            actual_score_a = 1.0 if comp['winner'] == comp['player_a'] else 0.0
            
            ratings[comp['player_a']] += 32 * (actual_score_a - expected_a)
            ratings[comp['player_b']] += 32 * ((1 - actual_score_a) - (1 - expected_a))
        
        # Sort by rating
        rankings = dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True))
        
        print("Final Elo Rankings:")
        for i, (player, rating) in enumerate(rankings.items(), 1):
            print(f"  {i:2d}. {player[:25]:25s}: {rating:7.1f}")
            
        return rankings
    
    def calculate_all_metrics(self, summary_data):
        """Calculate metrics for all summaries with GPU optimization"""
        print("Calculating metrics...")
        
        # Prepare data for batch processing
        references = [summary['original'] for summary in summary_data]
        candidates = [summary['text'] for summary in summary_data]
        players = [summary['player'] for summary in summary_data]
        
        metrics_data = []
        
        # Calculate non-BLEURT metrics first (these are fast)
        print("  Calculating BLEU, ROUGE-L, METEOR, Verbatim...")
        for i, (ref, cand, player) in enumerate(zip(references, candidates, players)):
            if i % 1000 == 0:
                print(f"    Processed {i}/{len(summary_data)}")
            
            # Calculate fast metrics
            row = {
                'player': player,
                'bleu': 0.0,
                'rouge_l': 0.0, 
                'meteor': 0.0,
                'verbatim': 0.0,
                'bleurt': 0.0
            }
            
            try:
                # BLEU
                ref_tokens = ref.split()
                cand_tokens = cand.split()
                if len(cand_tokens) > 0:
                    row['bleu'] = sentence_bleu([ref_tokens], cand_tokens)
                
                # ROUGE-L
                rouge_scores = self.rouge_scorer.score(ref, cand)
                row['rouge_l'] = rouge_scores['rougeL'].fmeasure
                
                # METEOR  
                row['meteor'] = meteor_score([ref.split()], cand.split())
                
                # Verbatim
                ref_words = set(ref.lower().split())
                cand_words = set(cand.lower().split())
                if len(ref_words) > 0:
                    row['verbatim'] = len(ref_words.intersection(cand_words)) / len(ref_words)
                    
            except Exception as e:
                pass  # Keep defaults
            
            metrics_data.append(row)
        
        # Calculate BLEURT in batches if available
        if self.bleurt_scorer is not None:
            print(f"  Calculating BLEURT in batches of {self.bleurt_batch_size}...")
            
            all_bleurt_scores = []
            for i in range(0, len(references), self.bleurt_batch_size):
                batch_refs = references[i:i + self.bleurt_batch_size]
                batch_cands = candidates[i:i + self.bleurt_batch_size]
                
                try:
                    batch_scores = self.bleurt_scorer.score(
                        references=batch_refs,
                        candidates=batch_cands
                    )
                    all_bleurt_scores.extend(batch_scores)
                except Exception as e:
                    print(f"    ⚠️ BLEURT batch failed: {e}")
                    all_bleurt_scores.extend([0.0] * len(batch_refs))
                
                if (i // self.bleurt_batch_size + 1) % 10 == 0:
                    print(f"    BLEURT batches processed: {i // self.bleurt_batch_size + 1}")
            
            # Update metrics_data with BLEURT scores
            for i, bleurt_score in enumerate(all_bleurt_scores):
                if i < len(metrics_data):
                    metrics_data[i]['bleurt'] = bleurt_score
        
        print(f"  ✅ Metrics calculated for {len(metrics_data)} summaries")
        return pd.DataFrame(metrics_data)
    
    def optimize_weights(self, player_metrics):
        """Find optimal weights"""
        print("Optimizing metric weights...")
        
        metric_cols = ['bleu', 'rouge_l', 'meteor', 'verbatim', 'bleurt']
        X = player_metrics[metric_cols].values
        y = player_metrics['elo_rating'].values
        
        def objective(weights):
            combined = X @ weights
            corr, _ = spearmanr(combined, y)
            return -corr if not np.isnan(corr) else 0
        
        # Optimize
        result = minimize(
            objective,
            np.ones(len(metric_cols)) / len(metric_cols),
            method='SLSQP',
            bounds=[(0, 1) for _ in metric_cols],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        if result.success:
            optimal_weights = result.x
            final_combined = X @ optimal_weights
            correlation, _ = spearmanr(final_combined, y)
            
            # Calculate individual Spearman correlations
            individual_corrs = []
            print(f"\nIndividual Spearman Correlations with Elo:")
            for metric in metric_cols:
                corr, p_value = spearmanr(player_metrics[metric], player_metrics['elo_rating'])
                individual_corrs.append(corr)
                print(f"  {metric.upper():10s}: {corr:7.4f} (p={p_value:.4f})")
            
            print(f"\nOptimization Results (Spearman Correlation):")
            print(f"  Combined metric correlation: {correlation:.4f}")
            print(f"  Optimal weights for linear combination:")
            for metric, weight in zip(metric_cols, optimal_weights):
                print(f"    {metric.upper():10s}: {weight:.4f}")
                
            return optimal_weights, metric_cols, correlation, individual_corrs
        
        return None, None, None, None
    
    def create_simple_visualization(self, player_metrics, weights, metric_cols, correlation):
        """Create simple visualization"""
        print("Creating visualization...")
        
        os.makedirs("datasets/summarize_feedback/simple_elo", exist_ok=True)
        
        # Calculate combined scores
        X = player_metrics[metric_cols].values
        combined_scores = X @ weights
        player_metrics['combined_score'] = combined_scores
        
        # Simple plot
        plt.figure(figsize=(12, 8))
        
        # Main correlation plot
        plt.subplot(2, 2, 1)
        plt.scatter(player_metrics['elo_rating'], player_metrics['combined_score'])
        plt.xlabel('Elo Rating')
        plt.ylabel('Combined Metric Score')
        plt.title(f'Elo vs Combined Score (ρ = {correlation:.3f})')
        
        # Weights plot
        plt.subplot(2, 2, 2)
        plt.bar(metric_cols, weights)
        plt.ylabel('Weight')
        plt.title('Optimal Weights')
        plt.xticks(rotation=45)
        
        # Individual correlations
        plt.subplot(2, 2, 3)
        individual_corrs = []
        for metric in metric_cols:
            corr, _ = spearmanr(player_metrics[metric], player_metrics['elo_rating'])
            individual_corrs.append(corr)
        
        plt.bar(metric_cols, individual_corrs)
        plt.ylabel('Correlation with Elo')
        plt.title('Individual Metric Correlations')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Top players
        plt.subplot(2, 2, 4)
        top_players = player_metrics.head(8)
        plt.barh(range(len(top_players)), top_players['elo_rating'])
        plt.yticks(range(len(top_players)), [p[:15] for p in top_players['player']])
        plt.xlabel('Elo Rating')
        plt.title('Top 8 Players')
        
        plt.tight_layout()
        plt.savefig('datasets/summarize_feedback/simple_elo/results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save results
        player_metrics.to_csv('datasets/summarize_feedback/simple_elo/player_metrics.csv', index=False)
        
        weights_df = pd.DataFrame({
            'metric': metric_cols,
            'weight': weights,
            'individual_correlation': individual_corrs
        })
        weights_df.to_csv('datasets/summarize_feedback/simple_elo/weights.csv', index=False)
        
        print("Results saved!")
    
    def print_final_results(self, player_metrics, weights, metric_cols, correlation, individual_corrs):
        """Print comprehensive final results to terminal"""
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY - ELO-BASED CORRELATION OPTIMIZATION")
        print("="*80)
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   • Number of players (models/humans): {len(player_metrics)}")
        print(f"   • Correlation method: Spearman")
        print(f"   • Metrics analyzed: {', '.join([m.upper() for m in metric_cols])}")
        print(f"   • Reference text: Original article/post (for metric calculation)")
        print(f"   • Note: 'ref' policy = human-written summary (competes as player)")
        
        print(f"\n🏆 TOP 5 PLAYERS BY ELO RATING:")
        for i, (_, row) in enumerate(player_metrics.head(5).iterrows(), 1):
            player_name = row['player']
            if player_name == 'ref':
                player_name = 'HUMAN-WRITTEN SUMMARY'
            elif 'sup4' in player_name:
                player_name = player_name.replace('sup4_', '').replace('ppo_rm3_kl', 'PPO-RM3-KL').replace('6b_', '6B-')
            elif 'pretrain' in player_name:
                player_name = 'PRETRAIN-6B'
            print(f"   {i}. {player_name[:35]:35s}: {row['elo_rating']:7.1f}")
        
        if weights is not None and individual_corrs is not None:
            print(f"\n📈 SPEARMAN CORRELATIONS WITH ELO RATINGS:")
            for metric, corr in zip(metric_cols, individual_corrs):
                print(f"   • {metric.upper():10s}: {corr:7.4f}")
            
            print(f"\n⚖️  OPTIMAL LINEAR COMBINATION:")
            print(f"   • Combined correlation: {correlation:.4f}")
            print(f"   • Optimal weights:")
            for metric, weight in zip(metric_cols, weights):
                print(f"     - {metric.upper():10s}: {weight:.4f}")
            
            # Calculate which metric contributes most
            max_weight_idx = np.argmax(weights)
            max_corr_idx = np.argmax([abs(c) for c in individual_corrs])
            
            print(f"\n🎯 KEY INSIGHTS:")
            print(f"   • Highest individual correlation: {metric_cols[max_corr_idx].upper()} ({individual_corrs[max_corr_idx]:.4f})")
            print(f"   • Highest weight in combination: {metric_cols[max_weight_idx].upper()} ({weights[max_weight_idx]:.4f})")
            
            if correlation > max(individual_corrs):
                improvement = correlation - max(individual_corrs)
                print(f"   • Linear combination improves best single metric by: {improvement:.4f}")
            else:
                print(f"   • Best single metric performs as well as combination")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"   • Player metrics: datasets/summarize_feedback/simple_elo/player_metrics.csv")
        print(f"   • Optimal weights: datasets/summarize_feedback/simple_elo/weights.csv")
        print(f"   • Visualization: datasets/summarize_feedback/simple_elo/results.png")
        
        print("\n" + "="*80)
    
    def run_analysis(self):
        """Run complete analysis"""
        print("=" * 50)
        print("ELO ANALYSIS - ALL 5 METRICS: BLEU, BLEURT, METEOR, ROUGE, VERBATIM")
        print("=" * 50)
        
        # Load data
        comparisons, summary_data = self.load_data(max_files=3)
        
        # Calculate Elo
        elo_rankings = self.calculate_elo_rankings(comparisons)
        
        # Calculate metrics
        metrics_df = self.calculate_all_metrics(summary_data)
        
        # Aggregate by player
        player_metrics = metrics_df.groupby('player').mean().reset_index()
        player_metrics['elo_rating'] = player_metrics['player'].map(elo_rankings)
        player_metrics = player_metrics.sort_values('elo_rating', ascending=False)
        
        print(f"\nPlayer metrics (top 10):")
        for _, row in player_metrics.head(10).iterrows():
            print(f"  {row['player'][:20]:20s}: Elo={row['elo_rating']:7.1f}, "
                  f"BLEU={row['bleu']:.3f}, ROUGE-L={row['rouge_l']:.3f}, "
                  f"METEOR={row['meteor']:.3f}, Verbatim={row['verbatim']:.3f}, BLEURT={row['bleurt']:.3f}")
        
        # Optimize weights
        weights, metric_cols, correlation, individual_corrs = self.optimize_weights(player_metrics)
        
        if weights is not None:
            # Visualize
            self.create_simple_visualization(player_metrics, weights, metric_cols, correlation)
        
        # Print comprehensive final results
        self.print_final_results(player_metrics, weights, metric_cols, correlation, individual_corrs)
        
        print("\nAnalysis complete!")
        return {
            'player_metrics': player_metrics,
            'weights': weights,
            'correlation': correlation,
            'individual_correlations': individual_corrs
        }

if __name__ == "__main__":
    analyzer = SimpleEloAnalyzer()
    results = analyzer.run_analysis()