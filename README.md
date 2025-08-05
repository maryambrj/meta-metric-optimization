# Meta-Metric Optimization and Auto-Elo Ranking

A comprehensive evaluation framework for text generation quality that combines traditional NLP metrics with Elo ranking systems to assess annotator quality and optimize metric combinations through linear regression.

## Project Overview

This framework evaluates text generation quality across multiple datasets using:
- **Traditional NLP Metrics**: BLEU, BLEURT, METEOR, ROUGE, Verbatim matching
- **Elo Ranking**: Competitive pairwise ranking system with proper Elo algorithm
- **Linear Combination Optimization**: Maximizes Spearman correlation between metrics and human preferences
- **GPU Acceleration**: Optimized for large-scale processing with A100 GPU support

## Supported Datasets

### 1. Summarize-from-Feedback Dataset (OpenAI) ⭐ **RECOMMENDED**
- **Size**: 64,832 summary comparisons from TL;DR dataset
- **Task**: Summarization quality evaluation with human preferences
- **Purpose**: Evaluate how well automated metrics align with human judgments in summarization
- **Features**: 
  - Original post text as reference
  - Two summaries per comparison (winner vs loser based on human preference)
  - Multiple model policies (supervised, RL, reward models)
  - Real-world Reddit TL;DR summarization scenarios

### 2. HH-RLHF Dataset (Anthropic)
- **Size**: Scalable to 161k samples
- **Task**: Human preference prediction (chosen vs rejected responses)
- **Purpose**: Large-scale evaluation of metric correlation with human judgments
- **Features**: Proper Elo ranking based on pairwise comparisons

### 3. Causal Relations Dataset
- **Size**: 20 samples, 10 annotators (7 humans + 3 LLMs)
- **Task**: Causal relation extraction from text
- **Purpose**: Evaluate annotator quality and metric effectiveness

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
- **Multi-Dataset Support**: Works across all datasets

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
│   ├── linear_optimization.py          # Linear combination optimization
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

### Dataset Selection Guide

#### 🎯 **For Summarization Evaluation** (Recommended)
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
   - Creates pairwise comparisons (winner vs loser)
   - Calculates Elo rankings where preferred response always wins
   - For summarize_feedback: Original post text becomes reference

2. **Metrics Step**:
   - Calculates BLEU, BLEURT, METEOR, ROUGE, Verbatim scores
   - For summarize_feedback: Compares both summaries against original post
   - Creates ranking tables for each metric

3. **Optimization Step**:
   - Finds optimal linear combination weights to maximize Spearman correlation with Elo
   - Uses cross-validation for robust optimization
   - Creates comprehensive visualizations and analysis

## Output Files Generated

### Core Analysis Files:
- **`linear_optimization_results.csv`**: Optimal weights and cross-validation results
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

## GPU Acceleration

### Performance Optimization
```bash
# Check GPU availability
python gpu_optimization.py

# Run with GPU acceleration
python core_scripts/calc_metrics.py --dataset summarize_feedback --batch_size 32
```

### Performance Benefits
- **BLEURT Calculation**: 5-10x faster with GPU
- **Batch Processing**: Efficient memory usage
- **Memory Management**: Automatic GPU memory cleanup

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
2. **Memory Issues**: Reduce `num_samples` or `batch_size` parameters
3. **GPU Issues**: Pipeline automatically falls back to CPU if GPU unavailable

### Performance Tips
- Use smaller `num_samples` for testing (100-1000)
- Increase `batch_size` for faster GPU processing (if memory allows)
- Use validation splits for final evaluation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.