# Summarize-from-Feedback Dataset

This directory contains the integration for the OpenAI summarize-from-feedback dataset, which provides human preference data for summarization tasks.

## Dataset Overview

The summarize-from-feedback dataset contains:
- **64,832 summary comparisons** on the TL;DR dataset
- **Human preference annotations** between pairs of summaries
- **Multiple model policies** including supervised, reward model, and RL fine-tuned versions
- **Structured comparison format** with winner/loser annotations

## Dataset Structure

Each sample contains:
- `info`: Original post information (title, text, subreddit)
- `summaries`: Array of two summaries to compare
- `choice`: Human preference (0 for first summary, 1 for second)
- `worker`: Annotator ID
- `batch`: Data collection batch
- `split`: Dataset split (train/valid1/valid2)

## Usage

### 1. Load Dataset

```bash
# Load training data (batches 3-10)
python core_scripts/summarize_feedback_loader.py --split train --num_samples 1000

# Load validation data
python core_scripts/summarize_feedback_loader.py --split valid2 --num_samples 500
```

### 2. Run Complete Pipeline

```bash
# Run all steps
python run_pipeline.py --dataset summarize_feedback --step all

# Run individual steps
python run_pipeline.py --dataset summarize_feedback --step data
python run_pipeline.py --dataset summarize_feedback --step metrics
python run_pipeline.py --dataset summarize_feedback --step optimization
```

### 3. Test Pipeline

```bash
# Run test suite with small sample
python test_summarize_feedback_pipeline.py
```

## Data Processing

The pipeline processes the data as follows:

1. **Download**: Fetches data from Azure Blob Storage
2. **Preprocess**: Extracts winner/loser pairs from comparisons
3. **Metrics**: Calculates BLEU, BLEURT, METEOR, ROUGE, and verbatim scores
   - **Reference**: Original post text (what summaries are trying to summarize)
   - **Candidates**: Winner and loser summaries (evaluated against the reference)
4. **Elo Rankings**: Creates Elo-style rankings based on human preferences
5. **Optimization**: Optimizes metric weights for correlation with human preferences

## Output Files

- `summarize_feedback_processed.csv`: Processed comparison data
- `final_elo_rankings.csv`: Elo rankings for all summaries
- `detailed_scores.csv`: Detailed metric scores
- `winner_annotations.csv`: Winner/loser annotations
- `rankings/`: Optimized metric combinations and correlation analysis

## Model Policies

The dataset includes summaries from various model policies:
- `ref`: Reference summaries
- `sup4`: Supervised baseline
- `ppo_xl`: RL fine-tuned policy
- `rm4`: Reward model
- And others...

## Key Differences from Other Datasets

1. **Summarization Focus**: Specifically designed for summarization quality evaluation
2. **Reddit Context**: Based on TL;DR dataset from Reddit posts
3. **Human Preferences**: Direct human judgments rather than automated metrics
4. **Policy Comparison**: Multiple model policies for comprehensive evaluation

## Citation

If you use this dataset, please cite:

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

## Troubleshooting

### Common Issues

1. **Download Failures**: Check internet connection and Azure Blob Storage availability
2. **Memory Issues**: Reduce `num_samples` or `batch_size` parameters
3. **GPU Issues**: The pipeline automatically falls back to CPU if GPU is unavailable

### Performance Tips

- Use smaller `num_samples` for testing (10-100)
- Increase `batch_size` for faster GPU processing
- Use `valid2` split for final evaluation (unseen during training)

## Evaluation Approach

For each data point:
- **Original post text** serves as the reference (what should be summarized)
- **Two summaries** are compared: winner (human preference) vs loser
- **Metrics are calculated** for both summaries against the reference
- **Correlation analysis** determines how well metrics align with human preferences

This approach allows us to evaluate:
- Which metrics best predict human preferences
- How well automated evaluation aligns with human judgment
- Optimal weight combinations for summarization evaluation

## Integration with Meta-Metric Optimization

This dataset is fully integrated with the meta-metric optimization pipeline:

1. **Metric Calculation**: All standard metrics (BLEU, BLEURT, METEOR, ROUGE, verbatim)
2. **Elo Rankings**: Human preference-based rankings
3. **Correlation Analysis**: Spearman correlation between metrics and human preferences
4. **Weight Optimization**: Linear combination optimization for best correlation

The summarize-feedback dataset provides a robust benchmark for evaluating how well automated metrics align with human judgments in summarization tasks. 