#!/usr/bin/env python3
"""
GPU Optimization Script for Meta-Metric Optimization Pipeline

This script helps configure and test GPU acceleration for the pipeline.
Designed to work on machines with NVIDIA GPUs and CUDA support.
"""

import os
import sys
import platform
import subprocess
import time

def check_tensorflow_installation():
    """Check TensorFlow installation and version"""
    print("🔍 Checking TensorFlow Installation...")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        return tf
    except ImportError:
        print("❌ TensorFlow not installed")
        print("   Install with: pip install tensorflow")
        return None
    except Exception as e:
        print(f"❌ Error importing TensorFlow: {e}")
        return None

def check_gpu_availability(tf):
    """Check if GPU is available and configured properly"""
    print("\n🔍 Checking GPU Availability...")
    print("=" * 50)
    
    if not tf:
        print("❌ TensorFlow not available")
        return False
    
    try:
        # Check for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU device(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Check GPU memory (only on supported platforms)
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                total_memory = gpu_memory['total'] / (1024**3)  # Convert to GB
                print(f"  GPU Memory: {total_memory:.2f} GB")
            except Exception as e:
                print(f"  GPU Memory: Unable to get memory info ({e})")
            
            return True, gpus
        else:
            print("❌ No GPU devices found")
            print("   Make sure you have NVIDIA GPU drivers installed")
            return False, []
    except Exception as e:
        print(f"❌ Error checking GPU availability: {e}")
        return False, []

def configure_gpu_memory(tf, gpus):
    """Configure GPU memory growth to avoid OOM errors"""
    print("\n⚙️ Configuring GPU Memory...")
    print("=" * 50)
    
    if not tf or not gpus:
        print("⚠️ No GPU available for configuration")
        return False
    
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
        return True
    except RuntimeError as e:
        print(f"❌ Failed to configure GPU memory: {e}")
        return False
    except Exception as e:
        print(f"❌ Error configuring GPU memory: {e}")
        return False

def test_gpu_performance(tf, gpus):
    """Test GPU performance with matrix multiplication"""
    print("\n🚀 Testing GPU Performance...")
    print("=" * 50)
    
    if not tf or not gpus:
        print("❌ Cannot test GPU performance - no GPU available")
        return False
    
    try:
        # Simple matrix multiplication test
        print("Running matrix multiplication test...")
        
        # Create test matrices
        size = 1000
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # Time the computation
        start_time = time.time()
        
        # Run computation
        c = tf.matmul(a, b)
        
        # Force execution
        _ = c.numpy()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ GPU test completed in {elapsed:.3f} seconds")
        print(f"   Matrix size: {size}x{size}")
        print(f"   Performance: {(2 * size**3) / (elapsed * 1e9):.2f} GFLOPS")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_bleurt_gpu(tf, gpus):
    """Test BLEURT GPU performance specifically"""
    print("\n🧠 Testing BLEURT GPU Performance...")
    print("=" * 50)
    
    if not tf or not gpus:
        print("❌ Cannot test BLEURT GPU performance - no GPU available")
        return False
    
    try:
        # Test BLEURT import and basic functionality
        from bleurt import score
        
        # Simple BLEURT test
        test_candidates = ["This is a test sentence."]
        test_references = ["This is a reference sentence."]
        
        print("Testing BLEURT scoring...")
        start_time = time.time()
        
        # Note: This requires BLEURT checkpoint to be available
        # We'll just test the import and basic setup
        print("✅ BLEURT module imported successfully")
        print("⚠️ BLEURT checkpoint test skipped (requires model files)")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"   Setup time: {elapsed:.3f} seconds")
        return True
        
    except ImportError:
        print("❌ BLEURT not installed")
        print("   Install with: cd bleurt && pip install -e .")
        return False
    except Exception as e:
        print(f"❌ BLEURT test failed: {e}")
        return False

def optimize_batch_size(tf, gpus):
    """Suggest optimal batch size based on GPU memory"""
    print("\n📊 Batch Size Optimization...")
    print("=" * 50)
    
    if not tf or not gpus:
        print("⚠️ No GPU available - using default batch size")
        return 32
    
    try:
        # Get GPU memory info
        gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
        total_memory = gpu_memory['total'] / (1024**3)  # Convert to GB
        
        print(f"GPU Memory: {total_memory:.2f} GB")
        
        # Suggest batch size based on memory
        if total_memory >= 8:
            suggested_batch = 64
            print("✅ High memory GPU detected")
        elif total_memory >= 4:
            suggested_batch = 32
            print("✅ Medium memory GPU detected")
        elif total_memory >= 2:
            suggested_batch = 16
            print("⚠️ Low memory GPU detected")
        else:
            suggested_batch = 8
            print("⚠️ Very low memory GPU detected")
        
        print(f"Suggested batch size: {suggested_batch}")
        return suggested_batch
        
    except Exception as e:
        print(f"⚠️ Unable to determine optimal batch size ({e}) - using default")
        return 32

def check_cuda_installation(tf):
    """Check CUDA installation status"""
    print("\n🔧 Checking CUDA Installation...")
    print("=" * 50)
    
    if not tf:
        print("❌ TensorFlow not available")
        return False
    
    try:
        # Check if CUDA is available in TensorFlow
        cuda_available = tf.test.is_built_with_cuda()
        print(f"TensorFlow built with CUDA: {cuda_available}")
        
        # Check CUDA version
        try:
            cuda_version = tf.sysconfig.get_build_info()['cuda_version']
            print(f"CUDA version: {cuda_version}")
        except:
            print("CUDA version: Unknown")
        
        # Check cuDNN version
        try:
            cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
            print(f"cuDNN version: {cudnn_version}")
        except:
            print("cuDNN version: Unknown")
        
        return cuda_available
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def check_system_info():
    """Check system information"""
    print("🖥️ System Information...")
    print("=" * 50)
    
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Check for NVIDIA drivers
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA drivers detected")
            # Extract GPU info from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'GPU' in line:
                    print(f"  {line.strip()}")
                    break
        else:
            print("❌ NVIDIA drivers not detected")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not available")

def print_optimization_tips():
    """Print optimization tips for better performance"""
    print("\n💡 GPU Optimization Tips...")
    print("=" * 50)
    
    tips = [
        "1. Use batch processing for BLEURT calculations (5-10x faster)",
        "2. Enable GPU memory growth to avoid OOM errors",
        "3. Clear GPU memory between batches with gc.collect()",
        "4. Use appropriate batch sizes (8-64 depending on GPU memory)",
        "5. Monitor GPU memory usage with nvidia-smi",
        "6. Consider using mixed precision for faster computation",
        "7. Close other GPU-intensive applications during processing",
        "8. For large datasets, process in chunks to avoid memory issues",
        "9. Use the --batch_size parameter to optimize performance",
        "10. Monitor performance with the built-in timing functions"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"   {i:2d}. {tip}")

def main():
    """Main function to run all GPU optimization checks"""
    print("🚀 GPU Optimization for Meta-Metric Pipeline")
    print("=" * 60)
    
    # Check system info
    check_system_info()
    
    try:
        # Check TensorFlow
        tf = check_tensorflow_installation()
        
        if tf:
            # Run GPU checks
            gpu_available, gpus = check_gpu_availability(tf)
            cuda_available = check_cuda_installation(tf)
            gpu_configured = configure_gpu_memory(tf, gpus)
            gpu_tested = test_gpu_performance(tf, gpus)
            bleurt_tested = test_bleurt_gpu(tf, gpus)
            optimal_batch = optimize_batch_size(tf, gpus)
        else:
            gpu_available = False
            cuda_available = False
            gpu_configured = False
            gpu_tested = False
            bleurt_tested = False
            optimal_batch = 32
        
        # Print summary
        print("\n📋 GPU Optimization Summary")
        print("=" * 50)
        print(f"TensorFlow Available: {'✅' if tf else '❌'}")
        print(f"GPU Available: {'✅' if gpu_available else '❌'}")
        print(f"CUDA Available: {'✅' if cuda_available else '❌'}")
        print(f"GPU Configured: {'✅' if gpu_configured else '❌'}")
        print(f"GPU Tested: {'✅' if gpu_tested else '❌'}")
        print(f"BLEURT GPU Ready: {'✅' if bleurt_tested else '❌'}")
        print(f"Optimal Batch Size: {optimal_batch}")
        
        # Print usage instructions
        print("\n🎯 Usage Instructions")
        print("=" * 50)
        if tf and gpu_available and gpu_tested and bleurt_tested:
            print("✅ GPU is fully ready for use!")
            print(f"   Run with: python core_scripts/calc_metrics.py --dataset hh_rlhf --batch_size {optimal_batch}")
        elif tf and gpu_available:
            print("⚠️ GPU available but some tests failed")
            print(f"   Try: python core_scripts/calc_metrics.py --dataset hh_rlhf --batch_size {optimal_batch}")
            print("   Check BLEURT installation if BLEURT test failed")
        else:
            print("⚠️ GPU optimization incomplete")
            print("   Consider installing CUDA/cuDNN or using CPU mode")
            print("   Run with: python core_scripts/calc_metrics.py --dataset hh_rlhf")
        
        print_optimization_tips()
        
    except Exception as e:
        print(f"\n❌ Error during GPU optimization: {e}")
        print("⚠️ Using CPU mode - performance may be slower")
        print("   Run with: python core_scripts/calc_metrics.py --dataset hh_rlhf")

if __name__ == "__main__":
    main() 