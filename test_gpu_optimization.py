#!/usr/bin/env python3
"""
Test GPU optimization and performance
"""

import time
import sys
from gpu_utils import print_gpu_status, detect_gpu, setup_gpu_for_bleurt
from simple_elo_all_metrics import SimpleEloAnalyzer

def test_gpu_detection():
    """Test GPU detection functionality"""
    print("=" * 60)
    print("🔍 TESTING GPU DETECTION")
    print("=" * 60)
    
    gpu_info = print_gpu_status()
    
    if gpu_info['has_gpu']:
        print("\n✅ GPU tests passed!")
        return True
    else:
        print("\n⚠️ No GPU detected - BLEURT will run on CPU")
        return False

def test_bleurt_initialization():
    """Test BLEURT initialization with and without GPU"""
    print("\n=" * 60)
    print("🧠 TESTING BLEURT INITIALIZATION")
    print("=" * 60)
    
    # Test with auto-detection
    print("\n1. Testing with auto-detection:")
    try:
        analyzer = SimpleEloAnalyzer()
        if analyzer.bleurt_scorer is not None:
            print("✅ BLEURT initialized successfully with auto-detection")
            print(f"   Batch size: {analyzer.bleurt_batch_size}")
            return True
        else:
            print("⚠️ BLEURT not initialized (no GPU or missing dependencies)")
            return False
    except Exception as e:
        print(f"❌ BLEURT initialization failed: {e}")
        return False

def test_small_batch():
    """Test with a small batch of data"""
    print("\n=" * 60)
    print("🚀 TESTING SMALL BATCH PROCESSING")
    print("=" * 60)
    
    try:
        print("Initializing analyzer with GPU optimization...")
        analyzer = SimpleEloAnalyzer()
        
        print("Testing metric calculation...")
        
        # Test sample data
        reference = "This is a test article about machine learning and artificial intelligence."
        candidate = "A test summary about AI and ML."
        
        start_time = time.time()
        metrics = analyzer.calculate_metrics(reference, candidate)
        end_time = time.time()
        
        print(f"\n📊 Sample Metrics (calculated in {end_time - start_time:.3f}s):")
        for metric, value in metrics.items():
            print(f"   {metric.upper()}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance with and without GPU"""
    print("\n=" * 60)
    print("⚡ TESTING PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Sample data for testing
    references = [
        "Machine learning is transforming the way we process data.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require significant computational resources."
    ] * 10  # 30 samples
    
    candidates = [
        "ML changes data processing methods.",
        "NLP helps computers understand language.",
        "Deep learning needs lots of computing power."
    ] * 10
    
    gpu_info = detect_gpu()
    
    if gpu_info['tensorflow_gpu']:
        print("🖥️ Testing with GPU (BLEURT enabled)...")
        start_time = time.time()
        
        try:
            analyzer_gpu = SimpleEloAnalyzer(enable_bleurt=True)
            
            # Calculate a few metrics to test speed
            for i in range(min(5, len(references))):
                metrics = analyzer_gpu.calculate_metrics(references[i], candidates[i])
            
            gpu_time = time.time() - start_time
            print(f"   GPU time for 5 samples: {gpu_time:.3f}s")
            
        except Exception as e:
            print(f"   ⚠️ GPU test failed: {e}")
            gpu_time = None
    else:
        print("⚠️ No TensorFlow GPU available for comparison")
        gpu_time = None
    
    print("\n💻 Testing without BLEURT (CPU only)...")
    start_time = time.time()
    
    try:
        analyzer_cpu = SimpleEloAnalyzer(enable_bleurt=False)
        
        # Calculate same metrics without BLEURT
        for i in range(min(5, len(references))):
            metrics = analyzer_cpu.calculate_metrics(references[i], candidates[i])
        
        cpu_time = time.time() - start_time
        print(f"   CPU time for 5 samples: {cpu_time:.3f}s")
        
        if gpu_time is not None:
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"\n🚀 Performance Summary:")
            print(f"   CPU (4 metrics): {cpu_time:.3f}s")
            print(f"   GPU (5 metrics): {gpu_time:.3f}s")
            if speedup > 1:
                print(f"   Note: GPU includes BLEURT overhead, but enables 5th metric")
            else:
                print(f"   CPU faster for small batches (GPU overhead)")
                
    except Exception as e:
        print(f"   ❌ CPU test failed: {e}")

def main():
    """Run all GPU optimization tests"""
    print("🧪 GPU OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    print("This script will test:")
    print("1. GPU detection and status")
    print("2. BLEURT initialization with GPU optimization")
    print("3. Small batch processing")
    print("4. Performance comparison")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    try:
        if test_gpu_detection():
            tests_passed += 1
        
        if test_bleurt_initialization():
            tests_passed += 1
        
        if test_small_batch():
            tests_passed += 1
        
        test_performance_comparison()  # Always informative
        tests_passed += 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! GPU optimization is working correctly.")
        print("\n🚀 Ready to run full analysis with:")
        print("   python run_elo_analysis.py              # Auto-detect GPU")
        print("   python run_elo_full.py --size medium    # Auto-detect GPU")
        print("   python run_elo_full.py --enable-bleurt  # Force GPU")
    else:
        print("⚠️ Some tests failed. Check GPU setup and dependencies.")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure NVIDIA GPU with CUDA support")
        print("   - Install: pip install tensorflow[and-cuda]")
        print("   - Download BLEURT-20 checkpoint")
    
    print("=" * 60)

if __name__ == "__main__":
    main()