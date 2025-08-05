# Meta-Metric Optimization and Auto-Elo Ranking

A comprehensive evaluation framework for text generation quality that combines traditional NLP metrics with Elo ranking systems to assess annotator quality and optimize metric combinations.

## Project Overview

This framework evaluates text generation quality across multiple datasets using:
- **Traditional NLP Metrics**: BLEU, BLEURT, METEOR, ROUGE, Verbatim matching
- **Elo Ranking**: Competitive pairwise ranking system with proper Elo algorithm
- **Linear Combination Optimization**: Maximizes Spearman correlation between metrics and human preferences
- **GPU Acceleration**: Optimized for large-scale processing with A100 GPU support

## Supported Datasets

### 1. Causal Relations Dataset
- **Size**: 20 samples, 10 annotators (7 humans + 3 LLMs)
- **Task**: Causal relation extraction from text
- **Purpose**: Evaluate annotator quality and metric effectiveness

### 2. HH-RLHF Dataset (Anthropic)
- **Size**: Scalable to 161k samples
- **Task**: Human preference prediction (chosen vs rejected responses)
- **Purpose**: Large-scale evaluation of metric correlation with human judgments
- **Features**: Proper Elo ranking based on pairwise comparisons

### 3. Summarize-from-Feedback Dataset (OpenAI) ⭐ **NEW**
- **Size**: 64,832 summary comparisons from TL;DR dataset
- **Task**: Summarization quality evaluation with human preferences
- **Purpose**: Evaluate how well automated metrics align with human judgments in summarization
- **Features**: 
  - Original post text as reference
  - Two summaries per comparison (winner vs loser based on human preference)
  - Multiple model policies (supervised, RL, reward models)
  - Real-world Reddit TL;DR summarization scenarios

## Key Features

### 🚀 GPU Acceleration
- **A100-80GB Support**: Optimized for large-scale processing
- **Batch Processing**: Efficient memory management
- **Performance Monitoring**: Real-time optimization feedback

### 🏆 Proper Elo Algorithm
- **Real Elo Calculations**: Not just fixed scores
- **Pairwise Comparisons**: Simulates actual competitive ranking
- **Dynamic Rating Updates**: Based on match outcomes

### 📊 Linear Combination Optimization
- **Spearman Correlation**: Maximizes correlation with human preferences
- **Cross-Validation**: Robust weight optimization
- **Multi-Dataset Support**: Works with both datasets

## Installation

### Quick Setup
```bash
# Run the automated setup script
./setup.sh
```

### Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install BLEURT from local directory
cd bleurt && pip install -e . && cd ..

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### GPU Acceleration (Optional)
```bash
# Check GPU availability and optimize settings
python gpu_optimization.py

# Test GPU optimization features
python test_gpu_optimization_fixed.py

# Run with GPU acceleration
python core_scripts/calc_metrics.py --dataset hh_rlhf --batch_size 32
```

### Requirements
- Python 3.7+
- pip
- Git (for cloning the repository)
- **Optional**: CUDA-compatible GPU for acceleration

## Project Structure

```
├── README.md                           # This file
├── setup.sh                           # Automated setup script
├── requirements.txt                   # Python dependencies
├── run_pipeline.py                    # Main pipeline script
├── config.py                          # Multi-dataset configuration
│
├── 📁 Core Scripts
│   ├── data_processing.py             # Data preprocessing (causal_relations)
│   ├── calc_metrics.py                # Metric calculation (both datasets)
│   ├── reg_test.py                    # Regression testing (both datasets)
│   ├── linear_optimization.py         # Linear combination optimization
│   ├── hh_rlhf_loader.py             # HH-RLHF dataset loader
│   └── test_hh_rlhf_pipeline.py      # Pipeline testing script
│
├── 📁 datasets/                       # Multi-dataset support
│   ├── 📁 causal_relations/           # Original causal relations dataset
│   │   ├── 📁 data/
│   │   │   ├── winner_annotations.csv
│   │   │   ├── final_elo_rankings.csv
│   │   │   └── detailed_scores.csv
│   │   ├── 📁 annotations/
│   │   │   ├── 📁 humans/             # Human annotator data
│   │   │   │   ├── Aadarsh.json
│   │   │   │   ├── Ashlin.json
│   │   │   │   ├── Kuldeep.json
│   │   │   │   ├── Maryam.json
│   │   │   │   ├── Nate.json
│   │   │   │   ├── Riley.json
│   │   │   │   └── Spencer.json
│   │   │   └── 📁 LLMs/               # LLM annotator data
│   │   │       ├── llama2_model.json
│   │   │       ├── llama3_model.json
│   │   │       └── mistral_model.json
│   │   ├── 📁 processed_data/         # Processed annotations
│   │   │   ├── Human1_elo_full.json
│   │   │   ├── Human2_elo_full.json
│   │   │   ├── ...
│   │   │   └── mistral_elo_full.json
│   │   └── 📁 rankings/               # Metric evaluation results
│   │       ├── bleu_values.csv
│   │       ├── bleurt_values.csv
│   │       ├── meteor_values.csv
│   │       ├── rouge_values.csv
│   │       ├── verbatim_values.csv
│   │       ├── elo_values.csv
│   │       ├── combined_metric_values.csv
│   │       ├── spearman_normalized_elo.csv
│   │       ├── spearman_per_sample.csv
│   │       └── bootstrapped_spearman_plot.png
│   │
│   └── 📁 hh_rlhf/                    # Anthropic HH-RLHF dataset
│       ├── 📁 data/
│       │   ├── hh_rlhf_processed.csv  # Processed HH-RLHF data
│       │   ├── winner_annotations.csv # Winner annotations
│       │   ├── final_elo_rankings.csv # Elo rankings
│       │   └── detailed_scores.csv    # Detailed metric scores
│       ├── 📁 annotations/
│       │   ├── chosen.json            # Chosen responses
│       │   └── rejected.json          # Rejected responses
│       ├── 📁 processed_data/         # Processed data
│       └── 📁 rankings/               # Metric evaluation results
│           ├── bleu_values.csv
│           ├── bleurt_values.csv
│           ├── meteor_values.csv
│           ├── rouge_values.csv
│           ├── verbatim_values.csv
│           ├── combined_metric_values.csv
│           └── bootstrapped_spearman_plot.png
│   │
│   └── 📁 summarize_feedback/         # OpenAI Summarize-from-Feedback dataset
│       ├── 📁 data/
│       │   ├── summarize_feedback_processed.csv  # Processed comparison data
│       │   ├── winner_annotations.csv            # Winner annotations
│       │   ├── final_elo_rankings.csv           # Elo rankings
│       │   └── detailed_scores.csv              # Detailed metric scores
│       ├── 📁 annotations/
│       │   ├── winner.json                      # Winner summaries
│       │   └── loser.json                       # Loser summaries
│       ├── 📁 processed_data/                   # Processed data
│       ├── 📁 rankings/                         # Metric evaluation results
│       │   ├── bleu_values.csv
│       │   ├── bleurt_values.csv
│       │   ├── meteor_values.csv
│       │   ├── rouge_values.csv
│       │   ├── verbatim_values.csv
│       │   ├── combined_metric_values.csv
│       │   └── bootstrapped_spearman_plot.png
│       └── README.md                           # Dataset-specific documentation
│
└── 📁 bleurt/                         # BLEURT evaluation framework
    ├── README.md
    ├── setup.py
    ├── bleurt/
    ├── BLEURT-20/
    └── ...
```

## Key Features

### 1. Multi-Annotator Evaluation
- **7 Human Annotators**: Aadarsh, Ashlin, Kuldeep, Maryam, Nate, Riley, Spencer
- **3 LLM Annotators**: Llama2, Llama3, Mistral
- **20 Text Samples**: Diverse causal relation extraction tasks

### 2. Summarization Evaluation (Summarize-from-Feedback) ⭐ **NEW**
- **Reference**: Original post text (what should be summarized)
- **Candidates**: Two summaries per comparison (winner vs loser based on human preference)
- **Evaluation**: Metrics calculated for both summaries against the reference
- **Correlation**: How well automated metrics align with human judgments
- **64,832 Comparisons**: Large-scale evaluation across multiple model policies

### 3. Comprehensive Metrics
- **BLEU**: N-gram overlap scoring
- **BLEURT**: Neural-based semantic similarity
- **METEOR**: Alignment-based evaluation
- **ROUGE**: Recall-oriented evaluation
- **Verbatim**: Exact string matching

### 4. Elo Ranking System
- Competitive pairwise ranking of annotators
- Dynamic score updates based on performance
- Quality assessment independent of traditional metrics

### 5. Meta-Metric Optimization
- Linear regression to find optimal metric weights
- Spearman correlation analysis
- Cross-validation with leave-samples-out approach


## Dataset Selection Guide

### Which Dataset Should You Use?

#### 🎯 **For Summarization Evaluation** (Recommended)
Use **Summarize-from-Feedback Dataset**:
- **Best for**: Evaluating summarization quality metrics
- **Why**: Specifically designed for summarization with human preferences
- **Data**: Real Reddit TL;DR posts with human-judged summaries
- **Size**: 64,832 comparisons across multiple model policies
- **Command**: `python run_pipeline.py --dataset summarize_feedback`

#### 🤖 **For General Text Generation Evaluation**
Use **HH-RLHF Dataset**:
- **Best for**: Evaluating general text generation quality
- **Why**: Large-scale human preference data for dialogue responses
- **Data**: Chosen vs rejected responses from human feedback
- **Size**: Scalable to 161k samples
- **Command**: `python run_pipeline.py --dataset hh_rlhf`

#### 🔬 **For Annotator Quality Assessment**
Use **Causal Relations Dataset**:
- **Best for**: Comparing human vs LLM annotator quality
- **Why**: Controlled environment with known ground truth
- **Data**: Causal relation extraction from 7 humans + 3 LLMs
- **Size**: 20 samples, 10 annotators
- **Command**: `python run_pipeline.py --dataset causal_relations`

## Usage

### Quick Start (Recommended)
```bash
# 1. Setup environment
./setup.sh

# 2. Run entire pipeline for causal relations dataset (default)
python run_pipeline.py --step all --dataset causal_relations

# 3. Or run HH-RLHF dataset
python run_pipeline.py --step all --dataset hh_rlhf

# 4. Or run Summarize-from-Feedback dataset (recommended for summarization evaluation)
python run_pipeline.py --step all --dataset summarize_feedback

# 5. Run specific steps for a dataset
python run_pipeline.py --step data --dataset hh_rlhf
python run_pipeline.py --step metrics --dataset hh_rlhf
python run_pipeline.py --step optimization --dataset hh_rlhf

# 6. Test the complete pipeline
python test_hh_rlhf_pipeline.py

# 6. Test comprehensive analysis (verifies all outputs match causal_relations)
python test_comprehensive_analysis.py
```

### Individual Script Execution

#### For Causal Relations Dataset
```bash
# Data Processing
cd core_scripts && python data_processing.py --dataset causal_relations

# Metric Calculation
cd core_scripts && python calc_metrics.py --dataset causal_relations

# Regression Testing
cd core_scripts && python reg_test.py --dataset causal_relations

# Linear Combination Optimization
cd core_scripts && python linear_optimization.py --dataset causal_relations
```

#### For HH-RLHF Dataset
```bash
# Download and process HH-RLHF dataset
cd core_scripts && python hh_rlhf_loader.py --num_samples 1000

# Calculate metrics for HH-RLHF (integrated into calc_metrics.py)
cd core_scripts && python calc_metrics.py --dataset hh_rlhf

# Regression Testing
cd core_scripts && python reg_test.py --dataset hh_rlhf

# Linear Combination Optimization
cd core_scripts && python linear_optimization.py --dataset hh_rlhf
```

#### For Summarize-from-Feedback Dataset ⭐ **NEW**
```bash
# Download and process summarize-feedback dataset
cd core_scripts && python summarize_feedback_loader.py --split train --num_samples 1000

# Calculate metrics for summarize-feedback (integrated into calc_metrics.py)
cd core_scripts && python calc_metrics.py --dataset summarize_feedback

# Regression Testing
cd core_scripts && python reg_test.py --dataset summarize_feedback

# Linear Combination Optimization
cd core_scripts && python linear_optimization.py --dataset summarize_feedback

# Test the complete pipeline
python test_summarize_feedback_pipeline.py
```

### Pipeline Steps Explained

#### For Causal Relations Dataset
1. **Data Step** (`--step data --dataset causal_relations`):
   - Processes annotation files from humans and LLMs
   - Creates structured dataset linking samples to winners
   - Output: `datasets/causal_relations/data/winner_annotations.csv`

2. **Metrics Step** (`--step metrics --dataset causal_relations`):
   - Computes BLEU, BLEURT, METEOR, ROUGE, and Verbatim scores
   - Creates ranking tables for each metric
   - Output: `datasets/causal_relations/rankings/*_values.csv` files

3. **Optimization Step** (`--step optimization --dataset causal_relations`):
   - Performs linear combination optimization to maximize Spearman correlation with Elo
   - Uses cross-validation to find optimal metric weights
   - Creates combined metric scores and visualizations
   - Output: `datasets/causal_relations/rankings/linear_optimization_results.csv`

#### For HH-RLHF Dataset
1. **Data Step** (`--step data --dataset hh_rlhf`):
   - Downloads Anthropic HH-RLHF dataset from Hugging Face
   - Processes chosen/rejected response pairs for pairwise comparisons
   - Creates Elo rankings where chosen responses are winners
   - Output: `datasets/hh_rlhf/data/hh_rlhf_processed.csv`

2. **Metrics Step** (`--step metrics --dataset hh_rlhf`):
   - Uses chosen responses as ground truth reference
   - Calculates BLEU, BLEURT, METEOR, ROUGE, and Verbatim scores
   - Compares both chosen and rejected responses against ground truth
   - Creates ranking tables for metric-based evaluation
   - Output: `datasets/hh_rlhf/rankings/*_values.csv` files

3. **Optimization Step** (`--step optimization --dataset hh_rlhf`):
   - Performs linear combination optimization to maximize Spearman correlation with Elo
   - Uses cross-validation to find optimal metric weights
   - Compares metric rankings with human preference rankings (Elo)
   - Creates comprehensive visualizations and analysis
   - Output: `datasets/hh_rlhf/rankings/linear_optimization_results.csv`

#### For Summarize-from-Feedback Dataset ⭐ **NEW**
1. **Data Step** (`--step data --dataset summarize_feedback`):
   - Downloads OpenAI summarize-from-feedback dataset from Azure Blob Storage
   - Processes summary comparison pairs (winner vs loser based on human preference)
   - Uses original post text as reference for metric evaluation
   - Creates Elo rankings where human-preferred summaries are winners
   - Output: `datasets/summarize_feedback/data/summarize_feedback_processed.csv`

2. **Metrics Step** (`--step metrics --dataset summarize_feedback`):
   - Uses original post text as reference (what should be summarized)
   - Calculates BLEU, BLEURT, METEOR, ROUGE, and Verbatim scores for both summaries
   - Compares winner and loser summaries against the reference text
   - Creates ranking tables for metric-based evaluation
   - Output: `datasets/summarize_feedback/rankings/*_values.csv` files

3. **Optimization Step** (`--step optimization --dataset summarize_feedback`):
   - Performs linear combination optimization to maximize Spearman correlation with human preferences
   - Uses cross-validation to find optimal metric weights for summarization evaluation
   - Compares metric rankings with human preference rankings (Elo)
   - Creates comprehensive visualizations and analysis
   - Output: `datasets/summarize_feedback/rankings/linear_optimization_results.csv`

## Output Files Generated

Both datasets produce the same comprehensive analysis outputs:

### Core Analysis Files:
- **`combined_metric_values.csv`**: Linear combination scores with optimal weights
- **`spearman_normalized_elo.csv`**: Spearman correlations for each metric
- **`linear_optimization_results.csv`**: Optimal weights and cross-validation results
- **`bootstrapped_spearman_plot.png`**: Comprehensive visualization with 4 subplots:
  - Spearman correlation vs optimal weights
  - Combined score vs Elo correlation scatter plot
  - Individual metric correlations
  - Optimal weight distribution pie chart

### Individual Metric Files:
- **`elo_values.csv`**: Elo rankings for all samples
- **`bleu_values.csv`**: BLEU scores
- **`bleurt_values.csv`**: BLEURT scores
- **`meteor_values.csv`**: METEOR scores
- **`rouge_values.csv`**: ROUGE scores
- **`verbatim_values.csv`**: Verbatim matching scores

### Additional Analysis:
- **`combined_annotator_specific.csv`**: Annotator-specific analysis (chosen vs rejected for HH-RLHF)
- **`spearman_per_sample.csv`**: Per-sample correlation analysis

### File Categorization

#### **Core Scripts** (`core_scripts/`)
- `data_processing.py`: Data preprocessing and winner extraction
- `calc_metrics.py`: Metric calculation (BLEU, BLEURT, METEOR, ROUGE, Verbatim)
- `reg_test.py`: Regression testing and ranking table creation
- `linear_optimization.py`: Linear combination optimization to maximize Spearman correlation with Elo
- `hh_rlhf_loader.py`: HH-RLHF dataset loading and processing
- `summarize_feedback_loader.py`: Summarize-from-feedback dataset loading and processing ⭐ **NEW**

#### **Data Files** (`data/`)
- `winner_annotations.csv`: Ground truth annotations
- `final_elo_rankings.csv`: Elo rankings for all annotators
- `detailed_scores.csv`: Detailed metric scores

#### **Annotations** (`annotations/`)
- **Human Annotators**: 7 JSON files with causal relation annotations
- **LLM Annotators**: 3 JSON files with model-generated annotations

#### **Processed Data** (`processed_data/`)
- Processed annotation files ready for metric calculation

#### **Rankings** (`rankings/`)
- Individual metric scores (BLEU, BLEURT, METEOR, ROUGE, Verbatim)
- Combined metric values with optimized weights
- Correlation analysis results
- Visualization plots

#### **BLEURT Framework** (`bleurt/`)
- Complete BLEURT evaluation framework
- Pre-trained models and checkpoints
- Documentation and setup files

## GPU Acceleration

### Performance Optimization
The pipeline now supports GPU acceleration for faster metric calculation, especially for large datasets like HH-RLHF.

#### GPU Configuration
```bash
# Check GPU availability and optimize settings
python gpu_optimization.py
```

#### Optimized Usage
```bash
# Run with GPU acceleration and optimal batch size
python core_scripts/calc_metrics.py --dataset hh_rlhf --batch_size 32

# For larger datasets, increase batch size (if GPU memory allows)
python core_scripts/calc_metrics.py --dataset hh_rlhf --batch_size 64
```

#### Performance Benefits
- **BLEURT Calculation**: 5-10x faster with GPU
- **Batch Processing**: Efficient memory usage
- **Memory Management**: Automatic GPU memory cleanup
- **Performance Monitoring**: Real-time progress tracking

#### GPU Requirements
- **CUDA-compatible GPU** (NVIDIA recommended)
- **TensorFlow with GPU support**
- **Adequate GPU memory** (4GB+ recommended for large batches)

#### Troubleshooting
- If you encounter OOM errors, reduce batch size
- Ensure CUDA drivers are up to date
- Close other GPU-intensive applications during processing
