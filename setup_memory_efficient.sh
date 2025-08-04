#!/bin/bash

echo "🚀 Memory-Efficient Setup for Meta-Metric Optimization"
echo "======================================================"

# Set environment variables for memory efficiency
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_ENABLE_ONEDNN_OPTS=0

# Reduce TensorFlow memory usage
export TF_MEMORY_ALLOCATION=0.8

echo "✅ Environment variables set for memory efficiency"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install BLEURT
echo "🧠 Installing BLEURT..."
cd bleurt && pip install -e . && cd ..

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Check if BLEURT checkpoint exists
if [ ! -d "bleurt/BLEURT-20" ]; then
    echo "⚠️ BLEURT-20 checkpoint not found"
    echo "📥 Downloading BLEURT-20 checkpoint..."
    
    cd bleurt
    if command -v wget &> /dev/null; then
        wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
    elif command -v curl &> /dev/null; then
        curl -O https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
    else
        echo "❌ Neither wget nor curl found. Please download BLEURT-20.zip manually"
        exit 1
    fi
    
    unzip BLEURT-20.zip
    rm BLEURT-20.zip
    cd ..
fi

echo "✅ Memory-efficient setup complete!"
echo ""
echo "💡 Usage tips:"
echo "   - Run with CPU mode: python run_pipeline.py --dataset causal_relations"
echo "   - Use smaller batch sizes: python core_scripts/calc_metrics.py --batch_size 8"
echo "   - Monitor memory usage during execution" 