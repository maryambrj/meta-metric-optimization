#!/usr/bin/env python3
"""
Metric Calculator for Summarization Dataset

Calculates NLP metrics (BLEU, ROUGE, METEOR, BLEURT, Verbatim) for each summary
by comparing it against the original text as ground truth.

For each data sample:
- Original text = ground truth reference
- Summary 1 = candidate 1 
- Summary 2 = candidate 2

Outputs metric scores for both summaries across all samples.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import glob
import argparse
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import tensorflow as tf

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class MetricCalculator:
    """Calculate multiple NLP metrics for summarization quality"""
    
    def __init__(self, use_gpu=True, batch_size=32, bleurt_checkpoint=None):
        """
        Initialize metric calculator
        
        Args:
            use_gpu (bool): Whether to use GPU for BLEURT
            batch_size (int): Batch size for GPU processing
            bleurt_checkpoint (str): Path to BLEURT checkpoint
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.bleurt_scorer = None
        self.bleurt_checkpoint_path = bleurt_checkpoint
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BLEURT if available
        self._initialize_bleurt()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        # Download punkt_tab (newer NLTK versions)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("📥 Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
        
        # Download punkt (fallback for older versions)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("📥 Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("📥 Downloading NLTK wordnet...")
            nltk.download('wordnet', quiet=True)
        
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print("📥 Downloading NLTK omw-1.4...")
            nltk.download('omw-1.4', quiet=True)
    
    def _initialize_bleurt(self):
        """Initialize BLEURT scorer"""
        try:
            # Add BLEURT library to Python path
            bleurt_lib_path = os.path.join(project_root, 'bleurt')
            if os.path.exists(bleurt_lib_path):
                sys.path.insert(0, bleurt_lib_path)
                print(f"✅ Added BLEURT library path: {bleurt_lib_path}")
            else:
                print(f"❌ BLEURT library not found at: {bleurt_lib_path}")
                return
            
            # Find BLEURT checkpoint
            if self.bleurt_checkpoint_path:
                if os.path.exists(self.bleurt_checkpoint_path):
                    bleurt_checkpoint = self.bleurt_checkpoint_path
                    print(f"✅ Using provided BLEURT checkpoint: {bleurt_checkpoint}")
                else:
                    print(f"❌ Provided BLEURT checkpoint not found: {self.bleurt_checkpoint_path}")
                    return
            else:
                # Look for BLEURT-20 checkpoint (recommended)
                checkpoint_paths = [
                    os.path.join(project_root, 'BLEURT-20'),
                    'BLEURT-20',
                    os.path.join(project_root, 'bleurt', 'BLEURT-20'),
                ]
                
                bleurt_checkpoint = None
                for path in checkpoint_paths:
                    if os.path.exists(path):
                        # Verify it's a valid checkpoint (has saved_model.pb or config files)
                        if (os.path.exists(os.path.join(path, 'saved_model.pb')) or 
                            os.path.exists(os.path.join(path, 'bleurt_config.json'))):
                            bleurt_checkpoint = path
                            print(f"✅ Found BLEURT checkpoint at: {bleurt_checkpoint}")
                            break
                
                if bleurt_checkpoint is None:
                    print("⚠️ BLEURT-20 checkpoint not found, BLEURT scores will be 0")
                    print("   Searched paths:")
                    for path in checkpoint_paths:
                        print(f"     - {path}")
                    return
            
            # Configure GPU
            if self.use_gpu:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    print(f"🚀 Found {len(gpus)} GPU(s), enabling GPU acceleration for BLEURT")
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    print("⚠️ No GPU found, using CPU for BLEURT")
                    self.use_gpu = False
            
            # Import and initialize BLEURT scorer
            from bleurt import score
            
            self.bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)
            print("✅ BLEURT scorer initialized successfully")
            
        except Exception as e:
            print(f"⚠️ BLEURT initialization failed: {e}")
            print("⚠️ Continuing without BLEURT scoring")
            self.bleurt_scorer = None
    
    def calculate_bleu(self, reference, candidate):
        """Calculate BLEU score"""
        try:
            # Check for empty strings
            if not reference.strip() or not candidate.strip():
                return 0.0
            
            # Tokenize
            ref_tokens = nltk.word_tokenize(reference.lower())
            cand_tokens = nltk.word_tokenize(candidate.lower())
            
            # Check for empty token lists
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
            return float(score)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0
    
    def calculate_meteor(self, reference, candidate):
        """Calculate METEOR score"""
        try:
            # Check for empty strings
            if not reference.strip() or not candidate.strip():
                return 0.0
            
            # Tokenize
            ref_tokens = nltk.word_tokenize(reference.lower())
            cand_tokens = nltk.word_tokenize(candidate.lower())
            
            # Check for empty token lists
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            score = meteor_score([ref_tokens], cand_tokens)
            return float(score)
        except Exception as e:
            print(f"METEOR calculation error: {e}")
            return 0.0
    
    def calculate_rouge(self, reference, candidate):
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_verbatim(self, reference, candidate):
        """Calculate verbatim overlap score"""
        try:
            # Simple word overlap
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            
            if len(ref_words) == 0:
                return 0.0
            
            overlap = len(ref_words.intersection(cand_words))
            score = overlap / len(ref_words)
            return score
        except:
            return 0.0
    
    def calculate_bleurt_batch(self, references, candidates):
        """Calculate BLEURT scores in batch"""
        if self.bleurt_scorer is None:
            return [0.0] * len(candidates)
        
        try:
            scores = self.bleurt_scorer.score(references=references, candidates=candidates)
            # Handle both numpy arrays and lists
            if hasattr(scores, 'tolist'):
                return scores.tolist()
            else:
                return list(scores)
        except Exception as e:
            print(f"⚠️ BLEURT batch calculation failed: {e}")
            return [0.0] * len(candidates)
    
    def process_sample(self, reference, summary1, summary2):
        """
        Process a single data sample
        
        Args:
            reference (str): Original text (ground truth)
            summary1 (str): First summary
            summary2 (str): Second summary
            
        Returns:
            dict: Metric scores for both summaries
        """
        results = {
            'summary1': {},
            'summary2': {}
        }
        
        # Calculate BLEU
        results['summary1']['bleu'] = self.calculate_bleu(reference, summary1)
        results['summary2']['bleu'] = self.calculate_bleu(reference, summary2)
        
        # Calculate METEOR
        results['summary1']['meteor'] = self.calculate_meteor(reference, summary1)
        results['summary2']['meteor'] = self.calculate_meteor(reference, summary2)
        
        # Calculate ROUGE
        rouge1 = self.calculate_rouge(reference, summary1)
        rouge2 = self.calculate_rouge(reference, summary2)
        results['summary1'].update(rouge1)
        results['summary2'].update(rouge2)
        
        # Calculate Verbatim
        results['summary1']['verbatim'] = self.calculate_verbatim(reference, summary1)
        results['summary2']['verbatim'] = self.calculate_verbatim(reference, summary2)
        
        return results
    
    def process_dataset(self, data, save_path=None):
        """
        Process entire dataset
        
        Args:
            data (list): List of data samples with 'reference', 'summary1', 'summary2'
            save_path (str): Path to save results
            
        Returns:
            pd.DataFrame: Results with metric scores
        """
        print(f"🚀 Processing {len(data):,} samples...")
        
        all_results = []
        
        # Prepare data for batch BLEURT processing
        references = []
        summary1_list = []
        summary2_list = []
        
        for i, sample in enumerate(data):
            references.extend([sample['reference'], sample['reference']])
            summary1_list.append(sample['summary1'])
            summary2_list.append(sample['summary2'])
        
        candidates = []
        for i in range(len(data)):
            candidates.append(summary1_list[i])
            candidates.append(summary2_list[i])
        
        # Calculate BLEURT in batches
        print("🧠 Calculating BLEURT scores...")
        bleurt_scores = []
        if self.bleurt_scorer is not None:
            for i in range(0, len(candidates), self.batch_size):
                batch_refs = references[i:i+self.batch_size]
                batch_cands = candidates[i:i+self.batch_size]
                batch_scores = self.calculate_bleurt_batch(batch_refs, batch_cands)
                bleurt_scores.extend(batch_scores)
                
                if (i // self.batch_size + 1) % 10 == 0:
                    print(f"   Processed {i+len(batch_cands):,} BLEURT comparisons...")
        else:
            bleurt_scores = [0.0] * len(candidates)
        
        print("📊 Calculating other metrics...")
        
        # Process each sample
        for i, sample in enumerate(data):
            # Get BLEURT scores for this sample
            bleurt_sum1 = bleurt_scores[i*2] if i*2 < len(bleurt_scores) else 0.0
            bleurt_sum2 = bleurt_scores[i*2+1] if i*2+1 < len(bleurt_scores) else 0.0
            
            # Calculate other metrics
            results = self.process_sample(
                sample['reference'], 
                sample['summary1'], 
                sample['summary2']
            )
            
            # Add BLEURT scores
            results['summary1']['bleurt'] = bleurt_sum1
            results['summary2']['bleurt'] = bleurt_sum2
            
            # Create result record
            result_row = {
                'sample_id': sample.get('sample_id', i),
                'post_id': sample.get('post_id', ''),
                'batch': sample.get('batch', ''),
                'reference_length': len(sample['reference'].split()),
                'summary1_length': len(sample['summary1'].split()),
                'summary2_length': len(sample['summary2'].split()),
                'summary1_policy': sample.get('summary1_policy', ''),
                'summary2_policy': sample.get('summary2_policy', ''),
            }
            
            # Add metric scores
            for metric in ['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL', 'verbatim', 'bleurt']:
                result_row[f'summary1_{metric}'] = results['summary1'].get(metric, 0.0)
                result_row[f'summary2_{metric}'] = results['summary2'].get(metric, 0.0)
            
            all_results.append(result_row)
            
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i+1:,} samples...")
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        if save_path:
            results_df.to_csv(save_path, index=False)
            print(f"💾 Saved results to {save_path}")
        
        return results_df


def load_summarization_data():
    """Load complete summarization dataset"""
    print("📁 Loading summarization dataset...")
    
    # Find all comparison files
    comparisons_dir = os.path.join('datasets', 'summarize_feedback', 'comparisons')
    json_files = glob.glob(os.path.join(comparisons_dir, '*.json'))
    
    print(f"📊 Found {len(json_files)} comparison files")
    
    all_data = []
    
    for file_path in sorted(json_files):
        filename = os.path.basename(file_path)
        print(f"   Processing {filename}...")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            file_data = []
            for line in lines:
                if line.strip():
                    try:
                        comparison = json.loads(line)
                        
                        # Extract data
                        info = comparison['info']
                        summaries = comparison['summaries']
                        
                        if len(summaries) >= 2:
                            sample = {
                                'sample_id': len(all_data),  # This will be unique
                                'post_id': info.get('id', f"post_{len(all_data)}"),
                                'reference': info['post'],  # Original text as ground truth
                                'summary1': summaries[0]['text'],
                                'summary2': summaries[1]['text'],
                                'summary1_policy': summaries[0]['policy'],
                                'summary2_policy': summaries[1]['policy'],
                                'batch': comparison.get('batch', filename),
                                'choice': comparison.get('choice', 0)
                            }
                            file_data.append(sample)
                            
                    except json.JSONDecodeError:
                        continue
            
            print(f"     Loaded {len(file_data):,} samples")
            all_data.extend(file_data)
            
        except Exception as e:
            print(f"     Error loading {filename}: {e}")
            continue
    
    print(f"✅ Total samples loaded: {len(all_data):,}")
    return all_data


def print_summary_statistics(results_df):
    """Print summary statistics of the results"""
    print(f"\n📊 METRIC CALCULATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(results_df):,}")
    print(f"   Total summaries analyzed: {len(results_df) * 2:,}")
    print(f"   Unique policies: {len(set(results_df['summary1_policy'].unique()) | set(results_df['summary2_policy'].unique()))}")
    
    print(f"\n📏 Length Statistics:")
    print(f"   Avg reference length: {results_df['reference_length'].mean():.1f} words")
    print(f"   Avg summary1 length: {results_df['summary1_length'].mean():.1f} words")
    print(f"   Avg summary2 length: {results_df['summary2_length'].mean():.1f} words")
    
    print(f"\n🎯 Metric Statistics (Summary 1):")
    metrics = ['bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL', 'verbatim', 'bleurt']
    for metric in metrics:
        col = f'summary1_{metric}'
        if col in results_df.columns:
            mean_score = results_df[col].mean()
            std_score = results_df[col].std()
            print(f"   {metric.upper():<8}: {mean_score:.4f} ± {std_score:.4f}")
    
    print(f"\n🎯 Metric Statistics (Summary 2):")
    for metric in metrics:
        col = f'summary2_{metric}'
        if col in results_df.columns:
            mean_score = results_df[col].mean()
            std_score = results_df[col].std()
            print(f"   {metric.upper():<8}: {mean_score:.4f} ± {std_score:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Calculate NLP metrics for summarization dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU processing")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples (for testing)")
    parser.add_argument("--output", default="metric_scores.csv", help="Output file path")
    parser.add_argument("--bleurt-checkpoint", type=str, help="Path to BLEURT checkpoint (recommended: 'BLEURT-20')")
    
    args = parser.parse_args()
    
    print("🚀 Summarization Metric Calculator")
    print("=" * 50)
    
    # Load data
    data = load_summarization_data()
    if not data:
        print("❌ No data loaded")
        return
    
    # Limit samples if specified
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"🔬 Limited to {len(data):,} samples for testing")
    
    # Initialize metric calculator
    use_gpu = not args.no_gpu
    calculator = MetricCalculator(
        use_gpu=use_gpu, 
        batch_size=args.batch_size, 
        bleurt_checkpoint=args.bleurt_checkpoint
    )
    
    # Process dataset
    start_time = time.time()
    results_df = calculator.process_dataset(data, save_path=args.output)
    end_time = time.time()
    
    # Print summary
    print_summary_statistics(results_df)
    
    print(f"\n⏱️ Processing Time: {end_time - start_time:.2f} seconds")
    print(f"📊 Average time per sample: {(end_time - start_time) / len(data):.4f} seconds")
    
    print(f"\n✅ Metric calculation complete!")
    print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()