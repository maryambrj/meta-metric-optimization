# Meta-Metric Optimization and Auto-Elo Ranking

A comprehensive evaluation framework for text generation quality that combines traditional NLP metrics with dual optimization methods: Elo-based correlation optimization for ranking data and pairwise logistic regression for binary preference data.

## Project Overview

This framework evaluates text generation quality across multiple datasets using:
- **Traditional NLP Metrics**: BLEU, BLEURT, METEOR, ROUGE, Verbatim matching
- **Dual Optimization Methods**: 
  - **Elo-based Ranking**: For datasets with multiple annotators (causal relations)
  - **Pairwise Logistic Regression**: For binary preference data (summarization, dialogue)
- **Automatic Method Selection**: Framework automatically chooses the appropriate method based on data type
- **GPU Acceleration**: Optimized for large-scale processing with A100 GPU support

## Analysis Methods

### 🆕 **Method 1: Elo-based Correlation Optimization** ⭐ **RECOMMENDED**
- **Approach**: Chess tournament-style analysis where models/humans are "players"
- **Input**: Preference comparisons between summaries  
- **Output**: Elo rankings + optimal metric combination
- **Correlation**: Spearman correlation optimization
- **Metrics**: BLEU, ROUGE-L, METEOR, Verbatim, BLEURT
- **Benefits**: 
  - Intuitive tournament ranking system
  - Handles binary preference data elegantly
  - Real-world interpretable Elo scores
  - Comprehensive correlation analysis

### **Method 2: Traditional Pipeline** (Original Implementation)
- **Approach**: Direct metric correlation with rankings
- **Methods**: Pairwise logistic regression or ranking correlation
- **Use case**: For comparison with traditional approaches

## Supported Datasets

### 1. Summarize-from-Feedback Dataset (OpenAI) ⭐ **RECOMMENDED**
- **Size**: 64,832 summary comparisons from TL;DR dataset
- **Task**: Summarization quality evaluation with human preferences
- **Method**: Pairwise Logistic Regression (binary preference optimization)
- **Purpose**: Optimize metric weights to predict winner/loser preferences
- **Features**: 
  - Original post text as reference
  - Two summaries per comparison (winner vs loser based on human preference)
  - Multiple model policies (supervised, RL, reward models)
  - Real-world Reddit TL;DR summarization scenarios

### 2. HH-RLHF Dataset (Anthropic)
- **Size**: Scalable to 161k samples
- **Task**: Human preference prediction (chosen vs rejected responses)
- **Method**: Pairwise Logistic Regression (binary preference optimization)
- **Purpose**: Large-scale evaluation of metric correlation with human judgments
- **Features**: Binary preference data optimized for preference prediction accuracy

### 3. Causal Relations Dataset
- **Size**: 20 samples, 10 annotators (7 humans + 3 LLMs)
- **Task**: Causal relation extraction from text
- **Method**: Elo-based Correlation Optimization (traditional ranking approach)
- **Purpose**: Evaluate annotator quality and metric effectiveness with multiple candidates per sample

## Key Features

### 🚀 GPU Acceleration & Auto-Detection
- **Automatic GPU Detection**: Detects NVIDIA GPUs, CUDA, TensorFlow GPU support
- **Auto-Enable BLEURT**: Automatically enables BLEURT when GPU is available
- **Intelligent Batch Sizing**: Optimizes batch sizes based on GPU memory (8GB-24GB+)
- **Memory Management**: TensorFlow GPU memory growth to prevent allocation issues
- **A100-80GB Support**: Optimized for large-scale processing with high-memory GPUs
- **Fallback Support**: Gracefully handles GPU failures and falls back to CPU

### 🏆 Proper Elo Algorithm
- **Real Elo Calculations**: Not just fixed scores
- **Pairwise Comparisons**: Simulates actual competitive ranking
- **Dynamic Rating Updates**: Based on match outcomes

### 📊 Dual Optimization Methods
- **Elo-based Correlation Optimization**: Maximizes Spearman correlation for ranking data (causal relations)
- **Pairwise Logistic Regression**: Maximizes preference prediction accuracy for binary data (summarization, dialogue)
- **Automatic Method Selection**: Framework chooses the best approach based on data structure
- **Cross-Validation**: Robust weight optimization for both methods

## Installation

### Quick Setup
```bash
# Run the automated setup script
./setup_memory_efficient.sh
```

### Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install BLEURT from local directory
cd bleurt && pip install -e . && cd ..

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Download BLEURT-20 checkpoint (if not already present)
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
rm BLEURT-20.zip
```

### Requirements
- Python 3.7+
- pip
- Git (for cloning the repository)
- **Optional**: CUDA-compatible GPU for acceleration

## Project Structure

```
├── README.md                           # This file
├── setup_memory_efficient.sh           # Automated setup script
├── requirements.txt                    # Python dependencies
├── run_pipeline.py                     # Main pipeline script
├── config.py                           # Multi-dataset configuration
│
├── 📁 core_scripts/
│   ├── calc_metrics.py                 # Metric calculation (all datasets)
│   ├── data_processing.py              # Data preprocessing (causal_relations)
│   ├── hh_rlhf_loader.py              # HH-RLHF dataset loader
│   ├── summarize_feedback_loader.py    # Summarize-feedback dataset loader
│   ├── linear_optimization.py          # Dual-method optimization (automatic selection)
│   ├── pairwise_logistic_regression.py # Pairwise logistic regression for binary preference data
│   ├── reg_test.py                     # Regression testing (all datasets)
│   └── verify_summarize_feedback_evaluation.py  # Verification script
│
├── 📁 datasets/                        # Multi-dataset support
│   ├── 📁 summarize_feedback/          # OpenAI Summarize-from-Feedback
│   │   ├── 📁 data/
│   │   │   ├── summarize_feedback_processed.csv
│   │   │   ├── final_elo_rankings.csv
│   │   │   └── detailed_scores.csv
│   │   ├── 📁 annotations/
│   │   │   ├── winner.json
│   │   │   └── loser.json
│   │   └── 📁 rankings/                # Analysis results
│   │       ├── *_values.csv           # Individual metrics
│   │       ├── combined_metric_values.csv
│   │       ├── linear_optimization_results.csv
│   │       └── bootstrapped_spearman_plot.png
│   │
│   ├── 📁 hh_rlhf/                     # Anthropic HH-RLHF
│   │   └── [similar structure]
│   │
│   └── 📁 causal_relations/            # Original causal relations
│       └── [similar structure]
│
├── 📁 test_*.py                        # Pipeline validation scripts
├── 📁 BLEURT-20/                       # BLEURT checkpoint
└── 📁 bleurt/                          # BLEURT framework
```

## Quick Start

### 🚀 **NEW: Elo-based Correlation Optimization** (Recommended)

#### **Complete Analysis - One Simple Command**

```bash
# Run complete analysis on full dataset with all 5 metrics
python run_full_analysis.py
```

**How it works:**
- 🔍 **Auto-detects GPU** and enables BLEURT automatically if available
- 📊 **Full dataset**: All ~180K comparisons (~360K summaries)
- 🎯 **All 5 metrics**: BLEU, ROUGE-L, METEOR, Verbatim, BLEURT
- ⚡ **Optimized**: GPU batch processing, intelligent memory management
- 🏆 **Chess tournament approach**: Each model/human is a "player" with Elo rankings
- 📊 **Metrics vs original text**: Summaries compared against original article/post
- 📈 **Spearman optimization**: Finds optimal linear combination of metrics
- 💾 **Complete results**: Terminal output + saved visualizations + CSV data

**Output:**
- Terminal: Complete analysis results with correlations and rankings
- Files: `datasets/summarize_feedback/simple_elo/`
  - `results.png` - Comprehensive visualization
  - `player_metrics.csv` - All player data and metrics
  - `weights.csv` - Optimal metric weights and correlations

**Example Terminal Output:**
```
🏆 TOP 5 PLAYERS BY ELO RATING:
   1. PPO-RM3-KL6                        :  1763.8
   2. 6B-ppo_rm4_6B-kl14                 :  1653.8
   3. HUMAN (reference)                  :  1639.6
   4. PPO-RM3-KL22                       :  1638.0
   5. PPO-RM3-KL9                        :  1590.7

📈 SPEARMAN CORRELATIONS WITH ELO RATINGS:
   • METEOR    :  0.5989  ⭐ Best predictor
   • ROUGE_L   :  0.5714
   • VERBATIM  :  0.5659
   • BLEU      :  0.5330

⚖️  OPTIMAL LINEAR COMBINATION:
   • Combined correlation: 0.5989
   • Key insight: METEOR alone performs as well as combination
```

### Dataset Selection Guide

#### 🎯 **For Summarization Evaluation** (Classic Pipeline)
```bash
# Complete pipeline for summarization evaluation
python run_pipeline.py --step all --dataset summarize_feedback
```

#### 🤖 **For General Text Generation Evaluation**
```bash
# Complete pipeline for dialogue/response evaluation
python run_pipeline.py --step all --dataset hh_rlhf
```

#### 🔬 **For Annotator Quality Assessment**
```bash
# Complete pipeline for annotator comparison
python run_pipeline.py --step all --dataset causal_relations
```

### Step-by-Step Pipeline

```bash
# 1. Setup environment
./setup_memory_efficient.sh

# 2. Run individual steps for any dataset
python run_pipeline.py --step data --dataset summarize_feedback
python run_pipeline.py --step metrics --dataset summarize_feedback
python run_pipeline.py --step optimization --dataset summarize_feedback

# 3. Or run everything at once
python run_pipeline.py --step all --dataset summarize_feedback

# 4. Test pipeline (optional)
python test_summarize_feedback_pipeline.py
```

### Individual Script Usage

#### Summarize-from-Feedback Dataset
```bash
# Download and process dataset
cd core_scripts && python summarize_feedback_loader.py --split train --num_samples 1000

# Calculate metrics
cd core_scripts && python calc_metrics.py --dataset summarize_feedback

# Regression testing
cd core_scripts && python reg_test.py --dataset summarize_feedback

# Linear combination optimization
cd core_scripts && python linear_optimization.py --dataset summarize_feedback
```

## Pipeline Steps Explained

### How It Works

1. **Data Step**: 
   - Downloads/processes the dataset
   - For binary preference data (summarize_feedback, hh_rlhf): Creates winner/loser pairs
   - For ranking data (causal_relations): Creates Elo rankings with multiple annotators
   - Automatic data structure detection

2. **Metrics Step**:
   - Calculates BLEU, BLEURT, METEOR, ROUGE, Verbatim scores
   - For summarize_feedback: Compares summaries against original post text (reference)
   - For causal_relations: Compares annotations against winner annotation (reference)
   - Creates ranking tables for each metric

3. **Optimization Step** (Automatic Method Selection):
   - **For Binary Preference Data**: Uses pairwise logistic regression to optimize preference prediction accuracy
   - **For Ranking Data**: Uses correlation optimization to maximize Spearman correlation with Elo rankings
   - Cross-validation for robust weight estimation
   - Creates method-specific visualizations and analysis

## Output Files Generated

### Core Analysis Files:

#### For Binary Preference Data (summarize_feedback, hh_rlhf):
- **`pairwise_logistic_results.csv`**: Optimal weights and cross-validation accuracy results
- **`pairwise_predictions.csv`**: Detailed prediction results with winner/loser scores
- **`combined_metric_values.csv`**: Combined metric scores for winner/loser pairs
- **`pairwise_logistic_plot.png`**: Comprehensive visualization with 4 subplots:
  - Optimal weights from logistic regression
  - Score difference distribution
  - Prediction accuracy vs score difference
  - Weight distribution pie chart

#### For Ranking Data (causal_relations):
- **`linear_optimization_results.csv`**: Optimal weights and cross-validation correlation results
- **`combined_metric_values.csv`**: Linear combination scores with optimal weights
- **`spearman_normalized_elo.csv`**: Spearman correlations for each metric
- **`bootstrapped_spearman_plot.png`**: Comprehensive visualization with 4 subplots:
  - Spearman correlation vs optimal weights
  - Combined score vs Elo correlation scatter plot
  - Individual metric correlations
  - Optimal weight distribution pie chart

### Individual Metric Files:
- **`*_values.csv`**: Individual metric scores (BLEU, BLEURT, METEOR, ROUGE, Verbatim)
- **`elo_values.csv`**: Elo rankings for all samples

## Optimization Methods Explained

### Method 1: Pairwise Logistic Regression (Binary Preference Data)
**Used for**: `summarize_feedback`, `hh_rlhf`

**Problem**: With only 2 candidates per sample (winner/loser), ranking correlation is either 1 or -1, making traditional correlation optimization ineffective.

**Solution**: Train a logistic regression that predicts the winner from the difference of metric vectors:
- Let `x1` and `x2` be the metric vectors (BLEU, METEOR, etc.) for the two candidates
- Define `delta_x = x_winner - x_loser`
- Fit logistic regression: `P(winner) = sigmoid(weights · delta_x)`
- Optimize weights to maximize preference prediction accuracy

**Benefits**: 
- Directly optimizes for preference prediction
- Handles binary comparison data effectively
- Cross-validation provides robust accuracy estimates

### Method 2: Elo-based Correlation Optimization (Ranking Data)
**Used for**: `causal_relations`

**Problem**: Multiple annotators (10 candidates per sample) require ranking-based evaluation.

**Solution**: Traditional correlation optimization:
- Calculate Elo rankings from pairwise comparisons
- Compute linear combination of metrics: `combo_score = w1×metric1 + w2×metric2 + ...`
- Rank samples by combo scores
- Optimize weights to maximize Spearman correlation between combo rankings and Elo rankings

**Benefits**:
- Handles multiple candidates per sample effectively
- Preserves ranking relationships
- Cross-validation provides robust correlation estimates

## GPU Acceleration

### Performance Optimization
```bash
# Check GPU availability
python gpu_optimization.py

# Run with GPU acceleration
python core_scripts/calc_metrics.py --dataset summarize_feedback --batch_size 32
```

### GPU Testing & Verification
```bash
# Test GPU detection and optimization
python test_gpu_optimization.py

# Check GPU status only
python gpu_utils.py
```

### Performance Benefits
- **BLEURT Calculation**: 5-10x faster with GPU for large batches
- **Automatic Optimization**: Batch sizes optimized for available GPU memory
- **Memory Management**: TensorFlow GPU memory growth and cleanup
- **Intelligent Fallback**: Continues without BLEURT if GPU unavailable

## Citations

### Summarize-from-Feedback Dataset
If you use the Summarize-from-Feedback dataset, please cite:

```bibtex
@article{stiennon2020learning,
  title={Learning to summarize from human feedback},
  author={Stiennon, Nisan and Ouyang, Long and Wu, Jeffrey and Ziegler, Daniel M and Lowe, Ryan and Voss, Chelsea and Radford, Alec and Amodei, Dario and Christiano, Paul F},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={3008--3021},
  year={2020}
}
```

### HH-RLHF Dataset
```bibtex
@misc{bai2022training,
  title={Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback},
  author={Yuntao Bai and Andy Jones and Kamal Ndousse and Amanda Askell and Anna Chen and Nova DasSarma and Dawn Drain and Stanislav Fort and Deep Ganguli and Tom Henighan and Nicholas Joseph and Saurav Kadavath and Jackson Kernion and Tom Conerly and Sheer El-Showk and Nelson Elhage and Zac Hatfield-Dodds and Danny Hernandez and Tristan Hume and Scott Johnston and Shauna Kravec and Liane Lovitt and Neel Nanda and Catherine Olsson and Dario Amodei and Tom Brown and Jack Clark and Sam McCandlish and Chris Olah and Ben Mann and Jared Kaplan},
  year={2022},
  eprint={2204.05862},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Troubleshooting

### Common Issues
1. **BLEURT Download Failures**: Ensure BLEURT-20 checkpoint is downloaded correctly
   - BLEURT is optional - analysis continues with 4 metrics if unavailable
2. **Memory Issues**: Reduce `num_samples` or `batch_size` parameters
3. **GPU Issues**: Pipeline automatically falls back to CPU if GPU unavailable
4. **Missing dependencies**: Run `pip install -r requirements.txt`

### Performance Tips
- Use smaller `num_samples` for testing (100-1000)
- Increase `batch_size` for faster GPU processing (if memory allows)  
- Use validation splits for final evaluation

### Expected Output Files
After running any analysis:
```
datasets/summarize_feedback/simple_elo/
├── results.png              # 4-panel visualization
├── player_metrics.csv       # All player data and metrics  
└── weights.csv              # Optimal weights and correlations
```

### Performance & Dataset Size Information

**Dataset Statistics:**
- **Total available**: 178,944 comparisons (~360K summaries)
- **Default script**: Uses 12,177 comparisons (~25K summaries, ~7% of data)
- **Reason for subset**: Full dataset takes 30-45 minutes vs 2-3 minutes

**Metric Performance:**
- **Fast metrics**: BLEU, ROUGE-L, METEOR, Verbatim (CPU, seconds per thousand)
- **Slow metric**: BLEURT (GPU required, minutes per thousand)

**Hardware Requirements:**
- **CPU analysis**: 4GB RAM, any modern CPU
- **BLEURT analysis**: NVIDIA GPU with 8GB+ VRAM, CUDA toolkit

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.