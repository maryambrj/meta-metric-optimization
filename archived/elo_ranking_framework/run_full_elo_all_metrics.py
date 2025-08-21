#!/usr/bin/env python3
"""
Run full Elo analysis with ALL metrics on more data
"""
from simple_elo_all_metrics import SimpleEloAnalyzer

if __name__ == "__main__":
    analyzer = SimpleEloAnalyzer()
    
    # Modify to use more files
    analyzer.load_data = lambda max_files=5: analyzer.load_data.__func__(analyzer, max_files)
    
    print("Running full analysis with 5 files...")
    results = analyzer.run_analysis()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY:")
    print("="*60)
    
    print(f"Final correlation with Elo: {results['correlation']:.4f}")
    print("Optimal weights for linear combination:")
    for metric, weight in zip(['BLEU', 'ROUGE-L', 'METEOR', 'Verbatim'], results['weights']):
        print(f"  {metric:10s}: {weight:.4f}")
    
    print("\nTop 8 players by Elo rating:")
    top_players = results['player_metrics'].head(8)
    for i, (_, row) in enumerate(top_players.iterrows(), 1):
        player = row['player']
        if player == 'ref':
            player = 'HUMAN'
        elif 'sup4' in player:
            player = player.replace('sup4_ppo_rm3_kl', 'PPO-RM3-KL')
            player = player.replace('sup4_6b_ppo_rm4_6b_kl', '6B-PPO-RM4-KL')
            player = player.replace('sup4_6b_t', '6B-T')
            player = player.replace('sup4_t', 'T')
        elif 'pretrain' in player:
            player = 'PRETRAIN-6B'
        
        print(f"  {i}. {player[:20]:20s}: Elo={row['elo_rating']:7.1f}")
    
    print(f"\nResults saved to: datasets/summarize_feedback/simple_elo/")
    print("Check results.png for visualization!")