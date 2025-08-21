# Archived Components Summary

This folder contains components that were moved from the main codebase to simplify the framework and focus on metric correlation optimization.

## What Was Archived

### 🏆 Auto-Elo Ranking Framework (`elo_ranking_framework/`)
Components related to the tournament-style Elo ranking system:

**Files moved:**
- `elo_all_metrics_analysis.py` - Comprehensive Elo analysis for all metrics
- `elo_summarization_analysis.py` - Elo analysis specific to summarization
- `run_elo_analysis.py` - Main Elo analysis runner
- `run_elo_full.py` - Full Elo analysis script
- `run_full_analysis.py` - Complete analysis with Elo rankings
- `run_full_elo_all_metrics.py` - Elo metrics analysis runner
- `run_full_elo_analysis.py` - Full Elo analysis runner
- `simple_elo_all_metrics.py` - Simplified Elo metrics calculation
- `test_comprehensive_analysis.py` - Tests for comprehensive analysis

**Data directories moved:**
- `datasets/summarize_feedback/elo_analysis/` - Elo analysis results
- `datasets/summarize_feedback/simple_elo/` - Simple Elo data

### 🎯 Logistic Regression Prediction (`logistic_regression_prediction/`)
Components for binary preference prediction using pairwise logistic regression:

**Files moved:**
- `pairwise_logistic_regression.py` - Core logistic regression implementation
- `linear_optimization_with_logistic.py` - Original linear optimization with both methods
- `hh_rlhf_loader.py` - HH-RLHF dataset loader
- `summarize_feedback_loader.py` - Summarize-feedback dataset loader
- `test_dual_method_pipeline.py` - Tests for dual method approach
- `test_summarize_feedback_pipeline.py` - Summarize feedback tests
- `test_hh_rlhf_pipeline.py` - HH-RLHF tests
- `test_multi_dataset.py` - Multi-dataset tests

### 📚 Documentation (`README_and_docs/`)
Original comprehensive documentation:

**Files moved:**
- `README.md` - Original detailed README with all features
- `IMPLEMENTATION_SUMMARY.md` - Implementation details and summary

## Why These Were Archived

1. **Simplification**: The main codebase now focuses on a single, clear purpose - metric correlation optimization
2. **Reduced Complexity**: Removed dual-method complexity and automatic method selection
3. **Dataset Focus**: Concentrated on the causal_relations dataset which uses correlation-based optimization
4. **Maintainability**: Easier to understand and maintain with fewer moving parts

## How to Restore Components

If you need any of these components:

1. **Individual files**: Copy specific files back to their original locations
2. **Full Auto-Elo system**: Move entire `elo_ranking_framework/` contents back to root
3. **Logistic regression**: Move `pairwise_logistic_regression.py` back to `core_scripts/`
4. **Dataset support**: Move loader scripts back to `core_scripts/` and update pipeline

## Original Functionality

The archived components provided:

- **Auto-Elo Rankings**: Tournament-style competitive rankings using Elo algorithm
- **Binary Preference Prediction**: Logistic regression for winner/loser classification
- **Multi-dataset Support**: HH-RLHF and Summarize-feedback datasets
- **Dual Optimization Methods**: Automatic selection between correlation and logistic regression
- **Comprehensive Analysis**: Full pipeline with multiple evaluation approaches

## Current Framework Focus

The cleaned framework now provides:
- **Single Method**: Correlation-based optimization only
- **Single Dataset**: Causal relations focus
- **Clear Purpose**: Optimal metric weight discovery for correlation maximization
- **Simplified Pipeline**: Linear workflow without conditional complexity