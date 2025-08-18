#!/usr/bin/env python3
"""
Run Elo analysis with different data size options
"""

import argparse
from simple_elo_all_metrics import SimpleEloAnalyzer
from gpu_utils import detect_gpu

def main():
    parser = argparse.ArgumentParser(description='Elo-based Correlation Optimization with GPU Auto-Detection')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'full'], 
                       default='medium',
                       help='Dataset size: small (3 files), medium (10 files), large (20 files), full (all files)')
    parser.add_argument('--enable-bleurt', action='store_true',
                       help='Force enable BLEURT (overrides auto-detection)')
    parser.add_argument('--disable-bleurt', action='store_true',
                       help='Force disable BLEURT (overrides auto-detection)')
    parser.add_argument('--no-auto-gpu', action='store_true',
                       help='Disable automatic GPU detection and setup')
    
    args = parser.parse_args()
    
    # Handle BLEURT enable/disable logic
    enable_bleurt = None  # Auto-detect by default
    if args.enable_bleurt and args.disable_bleurt:
        print("❌ Error: Cannot use both --enable-bleurt and --disable-bleurt")
        return
    elif args.enable_bleurt:
        enable_bleurt = True
    elif args.disable_bleurt:
        enable_bleurt = False
    
    # Map size to number of files
    size_map = {
        'small': 3,    # ~12K comparisons, ~24K summaries
        'medium': 10,  # ~40K comparisons, ~80K summaries  
        'large': 20,   # ~80K comparisons, ~160K summaries
        'full': None   # All files (~180K comparisons, ~360K summaries)
    }
    
    max_files = size_map[args.size]
    
    print("🚀 Elo-based Correlation Optimization Analysis")
    print("=" * 50)
    print(f"📊 Dataset size: {args.size}")
    print(f"📁 Files to process: {'All' if max_files is None else max_files}")
    
    # Show GPU status before initialization
    if not args.no_auto_gpu:
        gpu_info = detect_gpu()
        if enable_bleurt is None:
            if gpu_info['tensorflow_gpu']:
                print("🖥️  BLEURT: Auto-enabled (GPU detected)")
            else:
                print("⚡ BLEURT: Auto-disabled (no TensorFlow GPU)")
        elif enable_bleurt:
            print("🖥️  BLEURT: Force enabled")
        else:
            print("⚡ BLEURT: Force disabled")
    else:
        print("🔧 GPU auto-detection: Disabled")
        if enable_bleurt:
            print("🖥️  BLEURT: Enabled")
        else:
            print("⚡ BLEURT: Disabled")
    
    print("=" * 50)
    
    # Create analyzer with GPU optimization
    analyzer = SimpleEloAnalyzer(
        enable_bleurt=enable_bleurt,
        auto_gpu=not args.no_auto_gpu
    )
    
    # Override the load_data method to use specified number of files
    original_load = analyzer.load_data
    def load_with_size():
        return original_load(max_files=max_files)
    analyzer.load_data = load_with_size
    
    # Run analysis
    results = analyzer.run_analysis()
    
    print(f"\n✅ Analysis complete with {args.size} dataset!")
    return results

if __name__ == "__main__":
    results = main()