#!/usr/bin/env python3
"""
Elo Tournament System for Summarization Dataset

Processes the complete dataset and ranks all models/humans using 
chess-style Elo ratings based on pairwise preference comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import sys
import json
import glob
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import get_data_dir

class EloTournament:
    """Elo rating system for summarization tournament"""
    
    def __init__(self, initial_rating=1500, k_factor=32):
        """
        Initialize Elo tournament system
        
        Args:
            initial_rating (int): Starting Elo rating for all players
            k_factor (int): K-factor determines rating sensitivity to results
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings = defaultdict(lambda: initial_rating)
        self.match_history = []
        self.rating_history = defaultdict(list)
        
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for player A against player B"""
        diff = (rating_b - rating_a) / 400
        # Prevent overflow by clamping the difference
        diff = max(-10, min(10, diff))
        return 1 / (1 + 10**diff)
    
    def update_ratings(self, winner, loser):
        """
        Update ratings after a match
        
        Args:
            winner (str): Policy name of the winner
            loser (str): Policy name of the loser
        """
        # Get current ratings
        winner_rating = self.ratings[winner]
        loser_rating = self.ratings[loser]
        
        # Calculate expected scores
        winner_expected = self.expected_score(winner_rating, loser_rating)
        loser_expected = self.expected_score(loser_rating, winner_rating)
        
        # Actual scores (1 for win, 0 for loss)
        winner_actual = 1.0
        loser_actual = 0.0
        
        # Update ratings
        winner_new = winner_rating + self.k_factor * (winner_actual - winner_expected)
        loser_new = loser_rating + self.k_factor * (loser_actual - loser_expected)
        
        self.ratings[winner] = winner_new
        self.ratings[loser] = loser_new
        
        # Record match
        match_record = {
            'winner': winner,
            'loser': loser,
            'winner_rating_before': winner_rating,
            'loser_rating_before': loser_rating,
            'winner_rating_after': winner_new,
            'loser_rating_after': loser_new,
            'winner_expected': winner_expected,
            'rating_change_winner': winner_new - winner_rating,
            'rating_change_loser': loser_new - loser_rating
        }
        self.match_history.append(match_record)
        
        # Record rating history
        self.rating_history[winner].append(winner_new)
        self.rating_history[loser].append(loser_new)
    
    def process_tournament(self, matches_df):
        """
        Process all matches in the tournament
        
        Args:
            matches_df (pd.DataFrame): DataFrame with winner_policy and loser_policy columns
        """
        print(f"🏆 Processing {len(matches_df):,} matches...")
        
        for idx, row in matches_df.iterrows():
            winner = row['winner_policy']
            loser = row['loser_policy']
            self.update_ratings(winner, loser)
            
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1:,} matches...")
        
        print(f"✅ Tournament complete! Processed {len(matches_df):,} total matches")
    
    def get_final_rankings(self):
        """Get final Elo rankings sorted by rating"""
        rankings = []
        for player, rating in self.ratings.items():
            rankings.append({
                'player': player,
                'final_rating': rating,
                'matches_played': self.get_match_count(player),
                'wins': self.get_wins(player),
                'losses': self.get_losses(player)
            })
        
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('final_rating', ascending=False)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
        
        # Calculate win rate
        rankings_df['win_rate'] = rankings_df['wins'] / (rankings_df['wins'] + rankings_df['losses'])
        
        return rankings_df
    
    def get_match_count(self, player):
        """Get total matches played by a player"""
        count = 0
        for match in self.match_history:
            if match['winner'] == player or match['loser'] == player:
                count += 1
        return count
    
    def get_wins(self, player):
        """Get total wins by a player"""
        return sum(1 for match in self.match_history if match['winner'] == player)
    
    def get_losses(self, player):
        """Get total losses by a player"""
        return sum(1 for match in self.match_history if match['loser'] == player)
    
    def create_visualization(self, rankings_df, save_dir):
        """Create comprehensive tournament visualization"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Summarization Elo Tournament Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Final Elo Rankings (Top 15)
        top_15 = rankings_df.head(15)
        bars = ax1.barh(range(len(top_15)), top_15['final_rating'], alpha=0.8)
        ax1.set_yticks(range(len(top_15)))
        ax1.set_yticklabels(top_15['player'])
        ax1.set_xlabel('Elo Rating')
        ax1.set_title('Top 15 Player Rankings (Final Elo Ratings)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rating in zip(bars, top_15['final_rating']):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                    f'{rating:.0f}', va='center', fontweight='bold')
        
        # Plot 2: Matches Played vs Win Rate
        # Filter out players with very few matches for clarity
        active_players = rankings_df[rankings_df['matches_played'] >= 100]
        scatter = ax2.scatter(active_players['matches_played'], active_players['win_rate'], 
                            s=active_players['final_rating']/10 + 50, alpha=0.6, 
                            c=active_players['final_rating'], cmap='viridis')
        ax2.set_xlabel('Total Matches Played')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Matches Played vs Win Rate (Active Players, Color/Size = Elo Rating)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Elo Rating')
        
        # Add annotations for top players
        for _, row in rankings_df.head(8).iterrows():
            if row['matches_played'] >= 100:
                ax2.annotate(row['player'], 
                            (row['matches_played'], row['win_rate']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.8)
        
        # Plot 3: Rating Distribution
        ax3.hist(rankings_df['final_rating'], bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(self.initial_rating, color='red', linestyle='--', 
                   label=f'Initial Rating ({self.initial_rating})')
        ax3.set_xlabel('Final Elo Rating')
        ax3.set_ylabel('Number of Players')
        ax3.set_title('Distribution of Final Elo Ratings')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rating changes over time for top players
        top_8_players = rankings_df.head(8)['player'].tolist()
        
        for player in top_8_players:
            if player in self.rating_history:
                history = [self.initial_rating] + self.rating_history[player]
                ax4.plot(range(len(history)), history, marker='o', 
                        markersize=2, label=player, linewidth=2)
        
        ax4.set_xlabel('Match Number')
        ax4.set_ylabel('Elo Rating')
        ax4.set_title('Rating Evolution for Top 8 Players')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(save_dir, "elo_tournament_results.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved tournament visualization to {plot_file}")
        plt.close()
    
    def print_tournament_summary(self, rankings_df):
        """Print comprehensive tournament summary"""
        print(f"\n🏆 ELO TOURNAMENT RESULTS")
        print("=" * 80)
        
        print(f"\n📊 TOURNAMENT STATISTICS:")
        print(f"   Total Players: {len(rankings_df)}")
        print(f"   Total Matches: {len(self.match_history):,}")
        print(f"   Initial Rating: {self.initial_rating}")
        print(f"   K-Factor: {self.k_factor}")
        
        print(f"\n🥇 TOP 15 PLAYERS BY ELO RATING:")
        for _, row in rankings_df.head(15).iterrows():
            print(f"   {row['rank']:2d}. {row['player']:<25} : {row['final_rating']:7.1f} "
                  f"({row['wins']:4d}W-{row['losses']:4d}L, {row['win_rate']:.1%} win rate)")
        
        print(f"\n📈 RATING STATISTICS:")
        print(f"   Highest Rating: {rankings_df['final_rating'].max():.1f} ({rankings_df.iloc[0]['player']})")
        print(f"   Lowest Rating:  {rankings_df['final_rating'].min():.1f} ({rankings_df.iloc[-1]['player']})")
        print(f"   Average Rating: {rankings_df['final_rating'].mean():.1f}")
        print(f"   Rating Spread:  {rankings_df['final_rating'].max() - rankings_df['final_rating'].min():.1f}")
        
        print(f"\n🎮 ACTIVITY STATISTICS:")
        print(f"   Most Active Player: {rankings_df.loc[rankings_df['matches_played'].idxmax(), 'player']} "
              f"({rankings_df['matches_played'].max():,} matches)")
        print(f"   Average Matches per Player: {rankings_df['matches_played'].mean():.1f}")
        
        # Identify likely humans vs models
        print(f"\n🤖 PLAYER ANALYSIS:")
        human_indicators = ['ref', 'human']
        likely_humans = rankings_df[rankings_df['player'].str.lower().str.contains('|'.join(human_indicators), na=False)]
        likely_models = rankings_df[~rankings_df['player'].str.lower().str.contains('|'.join(human_indicators), na=False)]
        
        if not likely_humans.empty:
            ref_row = likely_humans.iloc[0]
            print(f"   Human Reference (ref): Rank {ref_row['rank']}, Rating {ref_row['final_rating']:.1f}")
            print(f"   Human Performance: {ref_row['wins']:,} wins, {ref_row['losses']:,} losses ({ref_row['win_rate']:.1%})")
        
        if not likely_models.empty:
            best_model = likely_models.iloc[0]
            print(f"   Best Model: {best_model['player']} (Rank {best_model['rank']}, Rating {best_model['final_rating']:.1f})")
            
        # Models better than human
        if not likely_humans.empty:
            ref_rating = likely_humans.iloc[0]['final_rating']
            better_models = likely_models[likely_models['final_rating'] > ref_rating]
            print(f"   Models beating human: {len(better_models)}/{len(likely_models)} ({len(better_models)/len(likely_models):.1%})")


def show_detailed_rankings(rankings_df):
    """Display detailed rankings of all players"""
    print(f"\n🏅 COMPLETE PLAYER RANKINGS (All {len(rankings_df)} Players):")
    print(f"{'Rank':<4} {'Player':<30} {'Rating':<10} {'Matches':<8} {'W-L':<12} {'Win%':<6}")
    print("-" * 80)
    
    for _, row in rankings_df.iterrows():
        record = f"{row['wins']}-{row['losses']}"
        win_pct = f"{row['win_rate']:.1%}"
        print(f"{row['rank']:<4} {row['player']:<30} {row['final_rating']:<10.1f} {row['matches_played']:<8} {record:<12} {win_pct:<6}")


def load_summarization_data():
    """Load complete summarization dataset from all batch files"""
    print("📁 Loading complete summarization dataset...")
    
    # Find all comparison files
    comparisons_dir = os.path.join('datasets', 'summarize_feedback', 'comparisons')
    json_files = glob.glob(os.path.join(comparisons_dir, '*.json'))
    
    print(f"📊 Found {len(json_files)} comparison files")
    
    all_comparisons = []
    
    for file_path in sorted(json_files):
        filename = os.path.basename(file_path)
        print(f"   Processing {filename}...")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            file_comparisons = []
            for line in lines:
                if line.strip():
                    try:
                        comparison = json.loads(line)
                        file_comparisons.append(comparison)
                    except json.JSONDecodeError:
                        continue
            
            print(f"     Loaded {len(file_comparisons):,} comparisons")
            all_comparisons.extend(file_comparisons)
            
        except Exception as e:
            print(f"     Error loading {filename}: {e}")
            continue
    
    print(f"🎯 Total comparisons loaded: {len(all_comparisons):,}")
    
    # Convert to tournament format
    tournament_data = []
    
    for comparison in all_comparisons:
        try:
            summaries = comparison['summaries']
            choice = comparison['choice']  # 0 or 1
            
            if len(summaries) == 2:
                winner_idx = choice
                loser_idx = 1 - choice
                
                winner_policy = summaries[winner_idx]['policy']
                loser_policy = summaries[loser_idx]['policy']
                
                tournament_data.append({
                    'winner_policy': winner_policy,
                    'loser_policy': loser_policy,
                    'post_id': comparison['info']['id'],
                    'batch': comparison.get('batch', 'unknown')
                })
        except KeyError:
            continue
    
    df = pd.DataFrame(tournament_data)
    print(f"✅ Processed {len(df):,} valid pairwise comparisons")
    print(f"📊 Unique players: {len(set(df['winner_policy'].unique()) | set(df['loser_policy'].unique()))}")
    
    return df


def run_tournament():
    """Run the complete tournament"""
    print("🚀 Summarization Elo Tournament System")
    print("=" * 60)
    
    # Load complete data
    data = load_summarization_data()
    if data is None or len(data) == 0:
        print("❌ No data loaded")
        return None
    
    # Initialize tournament
    tournament = EloTournament(initial_rating=1500, k_factor=32)
    
    # Process all matches
    tournament.process_tournament(data)
    
    # Get final rankings
    rankings = tournament.get_final_rankings()
    
    # Print results
    tournament.print_tournament_summary(rankings)
    
    # Save results
    save_dir = os.path.join('datasets', 'summarize_feedback', 'elo_tournament_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save rankings CSV
    rankings_file = os.path.join(save_dir, 'elo_tournament_rankings.csv')
    rankings.to_csv(rankings_file, index=False)
    print(f"\n💾 Saved rankings to {rankings_file}")
    
    # Save raw tournament data
    data_file = os.path.join(save_dir, 'tournament_data.csv')
    data.to_csv(data_file, index=False)
    print(f"💾 Saved tournament data to {data_file}")
    
    # Create visualization
    tournament.create_visualization(rankings, save_dir)
    
    print(f"\n✅ Elo tournament complete! Results saved to {save_dir}")
    return rankings


def show_results():
    """Display results from saved tournament data"""
    results_file = 'datasets/summarize_feedback/elo_tournament_results/elo_tournament_rankings.csv'
    
    if not os.path.exists(results_file):
        print("❌ Tournament results not found. Run with --run first.")
        return
    
    rankings_df = pd.read_csv(results_file)
    
    print("🏆 SUMMARIZATION ELO TOURNAMENT RESULTS")
    print("=" * 80)
    
    print(f"\n📊 TOURNAMENT OVERVIEW:")
    print(f"   Total Players: {len(rankings_df)}")
    print(f"   Total Matches: {rankings_df['matches_played'].sum() // 2:,}")
    print(f"   Rating Range: {rankings_df['final_rating'].min():.1f} to {rankings_df['final_rating'].max():.1f}")
    
    # Top performers
    print(f"\n🥇 TOP 10 PERFORMERS:")
    for _, row in rankings_df.head(10).iterrows():
        print(f"   {row['rank']:2d}. {row['player']:<25} : {row['final_rating']:7.1f} ({row['win_rate']:.1%} win rate)")
    
    # Human vs AI analysis
    ref_data = rankings_df[rankings_df['player'] == 'ref']
    if not ref_data.empty:
        ref_row = ref_data.iloc[0]
        models = rankings_df[rankings_df['player'] != 'ref']
        better_models = models[models['final_rating'] > ref_row['final_rating']]
        
        print(f"\n🤖 HUMAN vs AI ANALYSIS:")
        print(f"   Human Reference: Rank {ref_row['rank']}/{len(rankings_df)} (Rating: {ref_row['final_rating']:.1f})")
        print(f"   Models beating human: {len(better_models)}/{len(models)} ({len(better_models)/len(models):.1%})")
        print(f"   Best Model: {models.iloc[0]['player']} (Rating: {models.iloc[0]['final_rating']:.1f})")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Summarization Elo Tournament System")
    parser.add_argument("--run", action="store_true", help="Run complete tournament (processes all data)")
    parser.add_argument("--show", action="store_true", help="Show results from saved tournament")
    parser.add_argument("--full", action="store_true", help="Show detailed rankings of all players")
    
    args = parser.parse_args()
    
    if args.run:
        rankings = run_tournament()
        if rankings is not None and args.full:
            show_detailed_rankings(rankings)
    elif args.show:
        show_results()
        if args.full:
            results_file = 'datasets/summarize_feedback/elo_tournament_results/elo_tournament_rankings.csv'
            if os.path.exists(results_file):
                rankings_df = pd.read_csv(results_file)
                show_detailed_rankings(rankings_df)
    else:
        print("🏆 Summarization Elo Tournament System")
        print("\nUsage:")
        print("  --run       Run complete tournament (processes all ~179k matches)")
        print("  --show      Show summary results from saved tournament")
        print("  --full      Show detailed rankings of all players (use with --run or --show)")
        print("\nExamples:")
        print("  python elo_tournament.py --run")
        print("  python elo_tournament.py --show --full")


if __name__ == "__main__":
    main()