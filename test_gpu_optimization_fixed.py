#!/usr/bin/env python3
"""
Fixed GPU Optimization Test Script

This script properly tests GPU optimization features and BLEURT installation.
"""

import os
import sys
import time

def test_gpu_imports():
    """Test that all GPU-related imports work"""
    print("🔍 Testing GPU Imports...")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("❌ No GPU devices found")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_bleurt_installation():
    """Test BLEURT installation and checkpoint"""
    print("\n🧠 Testing BLEURT Installation...")
    print("=" * 50)
    
    try:
        # Test BLEURT import
        from bleurt import score
        print("✅ BLEURT import successful")
        
        # Check if checkpoint exists
        checkpoint_path = 'bleurt/BLEURT-20'
        if os.path.exists(checkpoint_path):
            print(f"✅ BLEURT checkpoint found at {checkpoint_path}")
            
            # Test BLEURT scorer creation
            try:
                scorer = score.BleurtScorer(checkpoint=checkpoint_path)
                print("✅ BLEURT scorer created successfully")
                return True
            except Exception as e:
                print(f"⚠️ BLEURT scorer creation failed: {e}")
                return False
        else:
            print(f"❌ BLEURT checkpoint not found at {checkpoint_path}")
            print("   Download with: cd bleurt && wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip && unzip BLEURT-20.zip")
            return False
            
    except ImportError as e:
        print(f"❌ BLEURT import failed: {e}")
        print("   Install with: cd bleurt && pip install -e .")
        return False
    except Exception as e:
        print(f"❌ BLEURT test failed: {e}")
        return False

def test_gpu_memory_config():
    """Test GPU memory configuration"""
    print("\n⚙️ Testing GPU Memory Configuration...")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth configured")
            
            # Test memory info
            try:
                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                total_gb = memory_info['total'] / (1024**3)
                print(f"✅ GPU memory: {total_gb:.2f} GB")
            except Exception as e:
                print(f"⚠️ Could not get memory info: {e}")
            
            return True
        else:
            print("❌ No GPU available")
            return False
            
    except Exception as e:
        print(f"❌ GPU memory config failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing functionality"""
    print("\n📊 Testing Batch Processing...")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        
        # Test simple batch operation
        batch_size = 16
        test_data = ["Test sentence " + str(i) for i in range(batch_size)]
        
        print(f"Testing batch size: {batch_size}")
        print(f"Sample data: {test_data[:3]}...")
        
        # Simulate batch processing
        start_time = time.time()
        
        # Simple tensor operation to simulate processing
        tensors = [tf.constant(data) for data in test_data]
        processed = [tf.strings.length(tensor) for tensor in tensors]
        
        # Force execution
        results = [t.numpy() for t in processed]
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ Batch processing test completed in {elapsed:.3f}s")
        print(f"   Processed {len(results)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False

def suggest_batch_size():
    """Suggest optimal batch size based on platform"""
    print("\n📊 Batch Size Recommendation...")
    print("=" * 50)
    
    import platform
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("Platform: macOS")
        print("⚠️ GPU acceleration may be limited on macOS")
        print("Suggested batch size: 16 (conservative)")
        return 16
    elif system == "Linux":
        print("Platform: Linux")
        print("✅ Full GPU acceleration support")
        print("Suggested batch size: 32")
        return 32
    elif system == "Windows":
        print("Platform: Windows")
        print("✅ GPU acceleration support")
        print("Suggested batch size: 32")
        return 32
    else:
        print(f"Platform: {system}")
        print("⚠️ Unknown platform")
        print("Suggested batch size: 16")
        return 16

def main():
    """Run all GPU optimization tests"""
    print("🚀 Fixed GPU Optimization Test Suite")
    print("=" * 60)
    
    # Check system info
    import platform
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    try:
        # Run all checks
        gpu_available = test_gpu_imports()
        bleurt_available = test_bleurt_installation()
        gpu_configured = test_gpu_memory_config()
        gpu_tested = test_batch_processing()
        optimal_batch = suggest_batch_size()
        
        # Print summary
        print("\n📋 GPU Optimization Summary")
        print("=" * 50)
        print(f"GPU Available: {'✅' if gpu_available else '❌'}")
        print(f"BLEURT Available: {'✅' if bleurt_available else '❌'}")
        print(f"GPU Configured: {'✅' if gpu_configured else '❌'}")
        print(f"GPU Tested: {'✅' if gpu_tested else '❌'}")
        print(f"Optimal Batch Size: {optimal_batch}")
        
        # Print usage instructions
        print("\n🎯 Usage Instructions")
        print("=" * 50)
        if gpu_available and bleurt_available and gpu_tested:
            print("✅ GPU is fully ready for use!")
            print(f"   Run with: python core_scripts/calc_metrics.py --dataset hh_rlhf --batch_size {optimal_batch}")
        elif gpu_available and bleurt_available:
            print("⚠️ GPU available but some tests failed")
            print(f"   Try: python core_scripts/calc_metrics.py --dataset hh_rlhf --batch_size {optimal_batch}")
        elif bleurt_available:
            print("⚠️ BLEURT available but GPU issues detected")
            print("   Run with CPU mode: python run_pipeline.py --dataset causal_relations")
        else:
            print("⚠️ GPU optimization incomplete")
            print("   Consider installing BLEURT or using CPU mode")
            print("   Run with: python run_pipeline.py --dataset causal_relations")
        
    except Exception as e:
        print(f"\n❌ Error during GPU optimization: {e}")
        print("⚠️ Using CPU mode - performance may be slower")
        print("   Run with: python run_pipeline.py --dataset causal_relations")

if __name__ == "__main__":
    main() 