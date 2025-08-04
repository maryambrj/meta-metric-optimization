#!/bin/bash

echo "🚀 Setting up Meta-Metric Optimization Environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install pip requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Install BLEURT from local directory
echo "🔧 Installing BLEURT from local directory..."
cd bleurt
pip install -e .

if [ $? -eq 0 ]; then
    echo "✅ BLEURT installed successfully"
else
    echo "❌ Failed to install BLEURT"
    exit 1
fi

cd ..

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

if [ $? -eq 0 ]; then
    echo "✅ NLTK data downloaded successfully"
else
    echo "⚠️  NLTK data download failed (may not be critical)"
fi

echo ""
echo "🎉 Setup complete! You can now run the pipeline:"
echo "   python run_pipeline.py --step all"
echo ""
echo "Or run individual steps:"
echo "   python run_pipeline.py --step data"
echo "   python run_pipeline.py --step metrics"
echo "   python run_pipeline.py --step optimization" 