# Implementation Summary: Dual-Method Optimization

## Overview
Successfully implemented a dual-method optimization approach that automatically selects the appropriate optimization method based on data type:

- **Causal Relations**: Uses original Elo-based correlation optimization
- **Summary Data (summarize_feedback, hh_rlhf)**: Uses new pairwise logistic regression

## Key Changes Made

### 1. Created Pairwise Logistic Regression Module
**File**: `core_scripts/pairwise_logistic_regression.py`
- Implements pairwise logistic regression for binary preference data
- Uses difference vectors: `delta_x = x_winner - x_loser`
- Trains logistic regression to predict preference from metric differences
- Includes cross-validation and accuracy evaluation
- Creates specialized visualizations for binary preference data

### 2. Enhanced Linear Optimization
**File**: `core_scripts/linear_optimization.py`
- Added automatic method selection based on dataset type
- **For causal_relations**: Keeps original Elo-based correlation optimization
- **For summary data**: Uses pairwise logistic regression
- Simplified code by removing redundant logic
- Clean separation between the two approaches

### 3. Updated Main Pipeline
**File**: `run_pipeline.py`
- Automatic method selection and reporting
- Updated output file checking for both methods
- Enhanced documentation and user feedback

### 4. Comprehensive Documentation
**File**: `README.md`
- Updated to explain dual-method approach
- Clear distinction between optimization methods
- Detailed explanation of when each method is used
- Updated output file documentation

## Technical Implementation Details

### For Causal Relations Data (Original Method)
- **Data Structure**: 10 candidates per sample, multiple annotators
- **Reference**: Winner annotation serves as ground truth
- **Optimization**: Maximize Spearman correlation between metric rankings and Elo rankings
- **Output**: Traditional correlation-based results

### For Summary Data (New Method)
- **Data Structure**: 2 candidates per sample (winner/loser pairs)
- **Reference**: Input text summary serves as ground truth for calculating metrics
- **Optimization**: Pairwise logistic regression on preference prediction
- **Mathematical Approach**: 
  - Calculate metrics for both winner and loser against reference
  - Create difference vector: `delta_x = x_winner - x_loser`
  - Train: `P(winner) = sigmoid(weights · delta_x)`
  - Optimize weights to maximize preference prediction accuracy
- **Output**: Preference prediction accuracy and specialized visualizations

## Usage Examples

### Causal Relations (Elo-based Correlation)
```bash
python run_pipeline.py --dataset causal_relations
# Uses correlation optimization automatically
```

### Summary Data (Pairwise Logistic Regression)
```bash
python run_pipeline.py --dataset summarize_feedback
python run_pipeline.py --dataset hh_rlhf
# Uses pairwise logistic regression automatically
```

## Output Files

### For Summary Data:
- `pairwise_logistic_results.csv`: Weights and CV accuracy
- `pairwise_predictions.csv`: Detailed prediction results
- `pairwise_logistic_plot.png`: 4-panel visualization

### For Causal Relations:
- `linear_optimization_results.csv`: Weights and CV correlation
- `spearman_normalized_elo.csv`: Individual metric correlations
- `bootstrapped_spearman_plot.png`: 4-panel visualization

## Benefits of This Approach

1. **Automatic Method Selection**: No manual intervention required
2. **Appropriate Optimization**: Each data type uses its optimal method
3. **Backward Compatibility**: Existing causal relations workflows unchanged
4. **Enhanced Accuracy**: Binary preference data now uses appropriate logistic regression
5. **Clean Architecture**: Clear separation between methods

## Key Insight

The fundamental issue was that with only 2 candidates per sample in summary data, ranking correlation is either 1 or -1, making correlation-based optimization ineffective. The pairwise logistic regression approach directly optimizes for what we actually care about: predicting which candidate is preferred based on their metric differences.

This provides a much more robust and meaningful optimization for binary preference data while preserving the original method for ranking data where it works well.