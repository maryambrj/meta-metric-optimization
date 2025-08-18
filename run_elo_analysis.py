#!/usr/bin/env python3
"""
Simple command to run Elo-based correlation optimization analysis
"""

from simple_elo_all_metrics import SimpleEloAnalyzer

def main():
    print("🚀 Starting Elo-based Correlation Optimization Analysis...")
    print("   Metrics: BLEU, ROUGE-L, METEOR, Verbatim (+ BLEURT if GPU available)")
    print("   Method: Chess tournament approach with Spearman correlation")
    print("   Dataset: OpenAI Summarize-from-Feedback")
    print("   GPU Auto-Detection: Enabled")
    print()
    
    # Run the analysis with GPU auto-detection
    analyzer = SimpleEloAnalyzer()
    results = analyzer.run_analysis()
    
    return results

if __name__ == "__main__":
    results = main()