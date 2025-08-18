#!/usr/bin/env python3
"""
GPU detection and optimization utilities
"""

import os
import subprocess
import sys

def detect_gpu():
    """Detect if GPU is available and return GPU info"""
    gpu_info = {
        'has_gpu': False,
        'gpu_type': None,
        'gpu_memory': None,
        'cuda_available': False,
        'tensorflow_gpu': False,
        'torch_gpu': False
    }
    
    try:
        # Check NVIDIA GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                gpu_name, memory = lines[0].split(', ')
                gpu_info['has_gpu'] = True
                gpu_info['gpu_type'] = gpu_name
                gpu_info['gpu_memory'] = f"{memory} MB"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Check CUDA availability
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        gpu_info['torch_gpu'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info['has_gpu'] = True
            if not gpu_info['gpu_type']:
                gpu_info['gpu_type'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info['tensorflow_gpu'] = len(gpus) > 0
        if len(gpus) > 0:
            gpu_info['has_gpu'] = True
            if not gpu_info['gpu_type']:
                gpu_info['gpu_type'] = f"TensorFlow GPU ({len(gpus)} devices)"
    except ImportError:
        pass
    
    return gpu_info

def setup_gpu_for_bleurt():
    """Setup optimal GPU configuration for BLEURT"""
    try:
        import tensorflow as tf
        
        # Enable GPU memory growth to avoid taking all GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit if needed (optional, for shared GPU environments)
            # tf.config.experimental.set_memory_limit(gpus[0], 8192)  # 8GB limit
            
            print(f"✅ TensorFlow GPU configured for BLEURT with {len(gpus)} GPU(s)")
            return True
    except Exception as e:
        print(f"⚠️ Failed to configure TensorFlow GPU: {e}")
    
    return False

def optimize_batch_size_for_gpu(gpu_memory_mb, default_batch_size=32):
    """Optimize batch size based on available GPU memory"""
    if not gpu_memory_mb:
        return default_batch_size
    
    try:
        memory = int(gpu_memory_mb.replace(' MB', ''))
        
        # Rough heuristic: BLEURT needs ~100MB per sample
        if memory >= 24000:  # 24GB+
            return 64
        elif memory >= 16000:  # 16GB+
            return 48
        elif memory >= 12000:  # 12GB+
            return 32
        elif memory >= 8000:   # 8GB+
            return 16
        else:                  # <8GB
            return 8
    except:
        return default_batch_size

def print_gpu_status():
    """Print detailed GPU status information"""
    gpu_info = detect_gpu()
    
    print("🖥️  GPU Status:")
    print("=" * 40)
    
    if gpu_info['has_gpu']:
        print(f"✅ GPU Available: {gpu_info['gpu_type']}")
        if gpu_info['gpu_memory']:
            print(f"💾 GPU Memory: {gpu_info['gpu_memory']}")
        print(f"🔥 CUDA Available: {'Yes' if gpu_info['cuda_available'] else 'No'}")
        print(f"🧠 TensorFlow GPU: {'Yes' if gpu_info['tensorflow_gpu'] else 'No'}")
        print(f"🔥 PyTorch GPU: {'Yes' if gpu_info['torch_gpu'] else 'No'}")
        
        if gpu_info['tensorflow_gpu']:
            print("🚀 BLEURT will use GPU acceleration")
        else:
            print("⚠️ BLEURT will fall back to CPU")
    else:
        print("❌ No GPU detected")
        print("⚠️ BLEURT will run on CPU (much slower)")
    
    print("=" * 40)
    return gpu_info

if __name__ == "__main__":
    gpu_info = print_gpu_status()
    
    if gpu_info['has_gpu']:
        print(f"\n🎯 Recommended batch size: {optimize_batch_size_for_gpu(gpu_info['gpu_memory'])}")
        
        if gpu_info['tensorflow_gpu']:
            setup_success = setup_gpu_for_bleurt()
            if setup_success:
                print("✅ GPU setup complete - ready for BLEURT")
            else:
                print("⚠️ GPU setup failed - check TensorFlow installation")