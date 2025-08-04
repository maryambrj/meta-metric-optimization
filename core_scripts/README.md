# Core Scripts

This directory contains the main Python scripts for the Meta-Metric Optimization pipeline.

## Scripts Overview

### 1. `data_processing.py`
- **Purpose**: Processes annotation data and creates winner annotations CSV
- **Input**: JSON annotation files from humans and LLMs
- **Output**: `winner_annotations.csv` with structured data linking samples to winners
- **Usage**: `python data_processing.py`

### 2. `calc_metrics.py`
- **Purpose**: Computes multiple evaluation metrics (BLEU, BLEURT, METEOR, ROUGE, Verbatim) for both datasets
- **Input**: Dataset-specific annotation data and structures
- **Output**: Metric scores for each annotator and sample
- **Usage**: 
  - `python calc_metrics.py --dataset causal_relations`
  - `python calc_metrics.py --dataset hh_rlhf`

### 3. `reg_test.py`
- **Purpose**: Processes detailed scores and creates ranking tables with Spearman correlations for both datasets
- **Input**: Dataset-specific `detailed_scores.csv` and Elo rankings
- **Output**: Individual metric CSV files and Spearman correlation analysis
- **Usage**: 
  - `python reg_test.py --dataset causal_relations`
  - `python reg_test.py --dataset hh_rlhf`

### 4. `linear_regression_optimization.py`
- **Purpose**: Performs meta-metric optimization using linear regression with cross-validation for both datasets
- **Input**: Dataset-specific metric values and Elo rankings
- **Output**: Optimized metric weights, combined scores, and correlation analysis
- **Usage**: 
  - `python linear_regression_optimization.py --dataset causal_relations`
  - `python linear_regression_optimization.py --dataset hh_rlhf`

### 5. `hh_rlhf_loader.py`
- **Purpose**: Downloads and processes the Anthropic HH-RLHF dataset from Hugging Face
- **Input**: HH-RLHF dataset from Hugging Face Hub
- **Output**: Processed dataset with chosen/rejected pairs and Elo rankings
- **Usage**: `python hh_rlhf_loader.py --num_samples 1000`

## Pipeline Integration

These scripts are automatically called by the main pipeline (`run_pipeline.py`):

### For Causal Relations Dataset:
1. **Data Step**: Runs `data_processing.py --dataset causal_relations`
2. **Metrics Step**: Runs `calc_metrics.py --dataset causal_relations` followed by `reg_test.py --dataset causal_relations`
3. **Optimization Step**: Runs `linear_regression_optimization.py --dataset causal_relations`

### For HH-RLHF Dataset:
1. **Data Step**: Runs `hh_rlhf_loader.py --dataset_name hh_rlhf`
2. **Metrics Step**: Runs `calc_metrics.py --dataset hh_rlhf` followed by `reg_test.py --dataset hh_rlhf`
3. **Optimization Step**: Runs `linear_regression_optimization.py --dataset hh_rlhf`

## Dependencies

All scripts require the following packages (see `requirements.txt`):
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- tensorflow (for BLEURT)
- nltk (for METEOR)
- rouge-score

## Configuration

All scripts use the configuration from `config.py` in the project root for file paths and settings. 