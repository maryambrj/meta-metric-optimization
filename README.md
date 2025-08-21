# Metric Correlation Optimization Framework

A framework for optimizing linear combinations of NLP metrics to best correlate with human annotations using Spearman correlation optimization.

## Overview

This framework evaluates text generation quality by finding optimal weights for traditional NLP metrics (BLEU, BLEURT, METEOR, ROUGE, Verbatim) to maximize correlation with ground truth annotations.

## Supported Dataset

- **Causal Relations Dataset**: 20 samples with 10 annotators (7 humans + 3 LLMs) for causal relation extraction evaluation

## Key Features

- **Correlation-based Optimization**: Maximizes Spearman correlation between metric combinations and ground truth
- **Cross-validation**: Robust weight estimation using Leave-One-Out CV for small datasets
- **Comprehensive Analysis**: Individual metric correlations and optimal weight visualization
- **GPU Acceleration**: Optimized BLEURT calculation with automatic fallback to CPU

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install BLEURT from local directory
cd bleurt && pip install -e . && cd ..

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Download BLEURT-20 checkpoint (if not already present)
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
rm BLEURT-20.zip
```

## Quick Start

### Complete Pipeline
```bash
# Run complete analysis pipeline
python run_pipeline.py --step all --dataset causal_relations
```

### Step-by-Step
```bash
# 1. Data preprocessing
python run_pipeline.py --step data --dataset causal_relations

# 2. Metric calculation
python run_pipeline.py --step metrics --dataset causal_relations

# 3. Optimization
python run_pipeline.py --step optimization --dataset causal_relations
```

### Individual Scripts
```bash
# Data processing
cd core_scripts && python data_processing.py

# Metric calculation
cd core_scripts && python calc_metrics.py --dataset causal_relations

# Regression testing
cd core_scripts && python reg_test.py --dataset causal_relations

# Correlation optimization
cd core_scripts && python metric_correlation_optimization.py --dataset causal_relations
```

## How It Works

1. **Data Processing**: Loads human and LLM annotations for causal relation extraction
2. **Metric Calculation**: Computes BLEU, BLEURT, METEOR, ROUGE, Verbatim scores
3. **Optimization**: Uses Spearman correlation optimization to find optimal metric weights
4. **Analysis**: Generates comprehensive visualizations and correlation reports

## Output Files

- **`linear_optimization_results.csv`**: Optimal weights and cross-validation results
- **`combined_metric_values.csv`**: Linear combination scores with optimal weights
- **`spearman_normalized_elo.csv`**: Individual metric correlations
- **`bootstrapped_spearman_plot.png`**: 4-panel visualization showing:
  - Spearman correlation vs optimal weights
  - Combined score vs Elo correlation scatter plot
  - Individual metric correlations
  - Optimal weight distribution

## Project Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── run_pipeline.py                     # Main pipeline script
├── config.py                           # Configuration
│
├── 📁 core_scripts/
│   ├── calc_metrics.py                 # Metric calculation
│   ├── data_processing.py              # Data preprocessing
│   ├── linear_optimization.py          # Pipeline compatibility wrapper
│   ├── metric_correlation_optimization.py # Core optimization logic
│   └── reg_test.py                     # Regression testing
│
├── 📁 datasets/causal_relations/       # Causal relations dataset
│   ├── 📁 annotations/                 # Human and LLM annotations
│   ├── 📁 data/                        # Processed data
│   └── 📁 rankings/                    # Analysis results
│
├── 📁 BLEURT-20/                       # BLEURT checkpoint
├── 📁 bleurt/                          # BLEURT framework
│
└── 📁 archived_components/             # Archived unused components
    ├── 📁 elo_ranking_framework/       # Auto-Elo ranking files
    ├── 📁 logistic_regression_prediction/ # Logistic regression components
    └── 📁 README_and_docs/             # Original documentation
```

## Archived Components

Unused components have been moved to `archived_components/` for reference:

- **`elo_ranking_framework/`**: Auto-Elo ranking system files
- **`logistic_regression_prediction/`**: Pairwise logistic regression for winner/loser prediction
- **`README_and_docs/`**: Original comprehensive documentation

## Requirements

- Python 3.7+
- pip
- **Optional**: CUDA-compatible GPU for BLEURT acceleration

## License

This project is licensed under the MIT License.




elo_tournament.py
- --run: Processes complete dataset (178,939 matches,
  80 players)
  - --show: Displays saved tournament results
  - --full: Shows detailed rankings of all players




metric_calculator.py
  # Basic usage (full dataset, GPU enabled)
  python metric_calculator.py

  # Test with limited samples
  python metric_calculator.py --max-samples 1000

  # Use larger batch size for better GPU utilization
  python metric_calculator.py --batch-size 64

  # Run on CPU only (if no GPU)
  python metric_calculator.py --no-gpu

  # Custom output file
  python metric_calculator.py --output my_metrics.csv

  📊 Output Format
  The script creates a CSV file with columns like:
  - sample_id, post_id, batch
  - reference_length, summary1_length, summary2_length
  - summary1_policy, summary2_policy
  - summary1_bleu, summary2_bleu
  - summary1_rouge1, summary2_rouge1
  - summary1_meteor, summary2_meteor
  - summary1_bleurt, summary2_bleurt
  - summary1_verbatim, summary2_verbatim