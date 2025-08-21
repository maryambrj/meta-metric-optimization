#!/usr/bin/env python3
"""
Run full Elo analysis on all summarization data
"""

from elo_summarization_analysis import SummarizationEloAnalyzer

if __name__ == "__main__":
    # Run analysis on all data
    analyzer = SummarizationEloAnalyzer()
    results = analyzer.run_full_analysis(max_files=None)  # Use all files
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS:")
    print("="*60)
    
    if results['correlation'] is not None:
        print(f"Final Spearman correlation: {results['correlation']:.4f}")
        print(f"Cross-validation: {results['cv_correlations']}")
        
        # Show top performers
        print("\nTop 5 models by Elo rating:")
        for i, (player, rating) in enumerate(list(results['elo_rankings'].items())[:5], 1):
            print(f"  {i}. {player}: {rating:.1f}")
            
        print("\nOptimal metric weights:")
        for metric, weight in zip(results['metric_cols'], results['optimal_weights']):
            print(f"  {metric}: {weight:.4f}")
    
    print("\nAnalysis complete! Check 'datasets/summarize_feedback/elo_analysis/' for detailed results.")