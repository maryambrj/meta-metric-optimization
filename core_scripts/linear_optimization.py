#!/usr/bin/env python3
"""
Linear optimization wrapper for metric correlation optimization.

This script provides compatibility with the existing pipeline while using
the cleaned metric correlation optimization functionality.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from metric_correlation_optimization import main as correlation_main

def main():
    """Main function that delegates to correlation-based optimization"""
    import argparse
    parser = argparse.ArgumentParser(description="Linear combination optimization")
    parser.add_argument("--dataset", choices=["causal_relations"], 
                       default="causal_relations", help="Dataset to process")
    args = parser.parse_args()
    
    if args.dataset != "causal_relations":
        print(f"❌ This optimization script only supports causal_relations dataset.")
        print(f"   Provided dataset: {args.dataset}")
        print(f"   Supported datasets: causal_relations")
        return
    
    # Call the correlation-based optimization
    correlation_main()

if __name__ == "__main__":
    main()