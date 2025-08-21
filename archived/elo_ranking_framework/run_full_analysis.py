#!/usr/bin/env python3
"""
Complete Elo-based analysis on the full summarization dataset
- Auto-detects GPU and enables BLEURT if available
- Uses full dataset (all comparison files)
- Calculates all 5 metrics: BLEU, ROUGE-L, METEOR, Verbatim, BLEURT
- Generates comprehensive results and visualizations
"""

from simple_elo_all_metrics import SimpleEloAnalyzer

def main():
    print("🚀 COMPLETE ELO-BASED ANALYSIS - FULL DATASET")
    print("=" * 60)
    print("📊 Dataset: OpenAI Summarize-from-Feedback (FULL)")
    print("🎯 Method: Chess tournament Elo rankings")
    print("📈 Metrics: BLEU, ROUGE-L, METEOR, Verbatim, BLEURT")
    print("🔍 GPU: Auto-detection enabled")
    print("⚖️  Optimization: Spearman correlation")
    print("=" * 60)
    print()
    
    # Create analyzer with GPU auto-detection
    analyzer = SimpleEloAnalyzer()
    
    # Override to use ALL files (full dataset)
    original_load = analyzer.load_data
    def load_full_dataset():
        return original_load(max_files=None)  # None = all files
    analyzer.load_data = load_full_dataset
    
    # Run complete analysis
    print("🎬 Starting full analysis...")
    results = analyzer.run_analysis()
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📁 Results saved to: datasets/summarize_feedback/simple_elo/")
    print("   • results.png - Comprehensive visualization")
    print("   • player_metrics.csv - All player data and metrics")
    print("   • weights.csv - Optimal weights and correlations")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()