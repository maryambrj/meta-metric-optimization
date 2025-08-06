import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bleurt import score
import json
import ast
import re
import time
from scipy.stats import spearmanr
import os
import sys
from tqdm import tqdm
import argparse
import gc

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from config import get_dataset_config, get_data_dir, get_annotations_dir, get_processed_data_dir, get_rankings_dir

# GPU Configuration for optimal performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL

# Enable GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus and USE_GPU:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth setting failed: {e}")
        USE_GPU = False
else:
    print("⚠️ Using CPU mode for BLEURT calculations")
    USE_GPU = False

# Initialize BLEURT scorer with GPU support and CPU fallback
try:
    if USE_GPU:
        scorer = score.BleurtScorer(checkpoint=BLEURT_CHECKPOINT)
        print("✅ BLEURT initialized with GPU support")
    else:
        # Force CPU mode for BLEURT
        with tf.device('/CPU:0'):
            scorer = score.BleurtScorer(checkpoint=BLEURT_CHECKPOINT)
        print("✅ BLEURT initialized with CPU support")
except Exception as e:
    print(f"❌ BLEURT initialization failed: {e}")
    print("⚠️ Continuing without BLEURT scoring")
    scorer = None

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("GPU memory info:", tf.config.experimental.get_memory_info('GPU:0') if gpus else "No GPU available")

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics_times = {}
    
    def start(self, operation):
        self.start_time = time.time()
        print(f"🚀 Starting {operation}...")
    
    def end(self, operation):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.metrics_times[operation] = elapsed
            print(f"✅ {operation} completed in {elapsed:.2f}s")
            self.start_time = None
    
    def print_summary(self):
        print("\n📊 Performance Summary:")
        for operation, elapsed in self.metrics_times.items():
            print(f"  {operation}: {elapsed:.2f}s")
        total_time = sum(self.metrics_times.values())
        print(f"  Total time: {total_time:.2f}s")

# Global performance monitor
perf_monitor = PerformanceMonitor()


class DataPreprocessor:
    def __init__(self):
        pass
    def extract_nodes_and_edges(self, train_json, generated_json):
        ground_truth_src_nodes = set()
        ground_truth_tgt_nodes = set()
        predicted_src_nodes = set()
        predicted_tgt_nodes = set()
        ground_truth_edges = []
        predicted_edges = []
        
        for item in train_json:
            if isinstance(item, list):
                if len(item) == 1 and isinstance(item[0], dict):
                    item = item[0]
                else:
                    raise ValueError("train_json contains an unexpected list structure: {}".format(item))
           
            if isinstance(item, str):
                item = json.loads(item)
            for relation in item['causal relations']:
                ground_truth_src_nodes.add(relation['src'])
                ground_truth_tgt_nodes.add(relation['tgt'])
                ground_truth_edges.append((relation['src'], relation['direction'].strip(), relation['tgt']))

        for item in generated_json:
            if isinstance(item, str):
                item = json.loads(item)
            for relation in item['causal relations']:
                predicted_src_nodes.add(relation['src'])
                predicted_tgt_nodes.add(relation['tgt'])
                predicted_edges.append((relation['src'], relation['direction'].strip(), relation['tgt']))

        return (list(predicted_src_nodes), list(predicted_tgt_nodes), 
                list(ground_truth_src_nodes), list(ground_truth_tgt_nodes), 
                predicted_edges, ground_truth_edges)


class Metrics:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_bleurt(self, candidates, references):
        """Optimized BLEURT computation with batch processing"""
        perf_monitor.start("BLEURT computation")
        
        # Check if BLEURT scorer is available
        if scorer is None:
            print("⚠️ BLEURT scorer not available, returning zeros")
            perf_monitor.end("BLEURT computation")
            return [0.0] * len(candidates)
        
        # Process in batches for better memory utilization
        scores = []
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]
            batch_references = references[i:i + self.batch_size]
            
            try:
                batch_scores = scorer.score(candidates=batch_candidates, references=batch_references)
                scores.extend(batch_scores)
            except Exception as e:
                print(f"⚠️ BLEURT batch failed: {e}, using zeros for batch")
                scores.extend([0.0] * len(batch_candidates))
            
            # Clear memory after each batch
            gc.collect()
            if USE_GPU and gpus:
                tf.keras.backend.clear_session()
        
        perf_monitor.end("BLEURT computation")
        return scores

    def compute_bleu(self, candidates, references):
        """Optimized BLEU computation"""
        perf_monitor.start("BLEU computation")
        scores = []
        
        for candidate, reference in zip(candidates, references):
            if len(candidate.split()) == 0:
                scores.append(0)
                continue
            score = sentence_bleu([reference.split()], candidate.split())
            scores.append(score)
        
        perf_monitor.end("BLEU computation")
        return scores

    def compute_meteor(self, candidates, references):
        """Optimized METEOR computation"""
        perf_monitor.start("METEOR computation")
        scores = []
        
        for candidate, reference in zip(candidates, references):
            if len(candidate.split()) == 0:
                scores.append(0)
                continue
            score = meteor_score([reference.split()], candidate.split())
            scores.append(score)
        
        perf_monitor.end("METEOR computation")
        return scores

    def compute_rouge(self, candidates, references):
        scores = []
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        for candidate, reference in zip(candidates, references):
            score = scorer.score(reference, candidate)['rouge1'].fmeasure
         
            scores.append(score)
        return scores

    def compute_verbatim(self, candidates, references):
        scores = np.array(candidates)==np.array(references)
        return scores.astype(int)


class ScoreCalculator:
    cache = {}
    def __init__(self, filenames, metrics, thresholds):
        self.filenames = filenames
        self.metrics = metrics
        self.thresholds = thresholds
        self.metric_computer = Metrics()
        self.data_preprocessor = DataPreprocessor()
    def compute_scores(self, nodes1, nodes2, metric):
        candidates = [pred for pred in nodes1 for _ in nodes2]
        references = [ref for _ in nodes1 for ref in nodes2]

        if metric == 'bleurt':
            scores = self.metric_computer.compute_bleurt(candidates, references)            
            
        elif metric == 'bleu':
            scores = self.metric_computer.compute_bleu(candidates, references)
 
        elif metric == 'meteor':
            scores = self.metric_computer.compute_meteor(candidates, references)
        
        elif metric == 'rouge':
            scores = self.metric_computer.compute_rouge(candidates, references)
        
        elif metric == 'verbatim':
            scores = self.metric_computer.compute_verbatim(candidates, references)
        else:
            raise ValueError(f'Unknown metric: {metric}')

        return scores
    
    def find_node_mappings(self, pred_nodes, gt_nodes, node_matrix):
        mappings = {}
        for i, pred_node in enumerate(pred_nodes):
            for j, gt_node in enumerate(gt_nodes):
                if node_matrix[i, j]:
                    if pred_node not in mappings:
                        mappings[pred_node] = []
                    mappings[pred_node].append(gt_node)
        return mappings

    def find_matching_edges(self, pred_edges, gt_edges, pred_to_gt_mappings):
        matched_edges = []
        matched_pred_edges=[]
        for src, dir, tgt in pred_edges:
            dir=dir.strip()
            if src in pred_to_gt_mappings and tgt in pred_to_gt_mappings:
                for mapped_src in pred_to_gt_mappings[src]:
                    for mapped_tgt in pred_to_gt_mappings[tgt]:
                        if (mapped_src, dir, mapped_tgt) in gt_edges and (src,dir,tgt) in pred_edges:
                            if (mapped_src, dir, mapped_tgt) not in matched_edges:
                                matched_edges.append((mapped_src, dir, mapped_tgt))
                            matched_pred_edges.append((src,dir,tgt))
                            
        return matched_pred_edges,matched_edges
    
    def calculate_f1_score(self, tp, fp, fn):
        if tp + fp == 0: 
            precision = 0 
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0 
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            return 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            return f1_score
    def calculate_scores(self):
        overall_scores = []
        detailed_scores = []
        
        for model, filename in self.filenames.items():

            if filename.endswith('.json'):
                with open(filename) as f:
                    data = json.load(f)
                    train_json, generated_json = data['train'], data['generated']

                    df = pd.DataFrame({'train': train_json, 'generated': generated_json})

            else:
                df = pd.read_csv(filename)
                train_json, generated_json = self.data_preprocessor.convert_df_to_json(df)

            for metric in self.metrics:
                threshold = self.thresholds[metric]
                row_f1_scores = []

                for row_idx, row in df.iterrows():
                    single_row_df = pd.DataFrame([row])
                  
                    if filename.endswith('.json'):
                        generated_json_row = [generated_json[row_idx]]
                        train_item = train_json[row_idx]
                        if isinstance(train_item, list):
                            train_json_row = [{
                                "text": generated_json_row[0]["text"] if "text" in generated_json_row[0] else "",
                                "causal relations": train_item
                            }]
                        else:
                            train_json_row = [train_item]
                    else:
                        train_json_row, generated_json_row = self.data_preprocessor.convert_df_to_json(single_row_df)

                    
                    predicted_nodes_src_row, predicted_nodes_tgt_row, ground_truth_nodes_src_row, ground_truth_nodes_tgt_row, predicted_edges_row, ground_truth_edges_row = self.data_preprocessor.extract_nodes_and_edges(train_json_row, generated_json_row)

                    if filename not in self.cache:
                        ScoreCalculator.cache[filename] = {}
                    if row_idx not in self.cache[filename]:
                        ScoreCalculator.cache[filename][row_idx] = {}
                    if metric not in self.cache[filename][row_idx]:
                        ScoreCalculator.cache[filename][row_idx][metric] = {}
                    if 'src' not in self.cache[filename][row_idx][metric]:
                        ScoreCalculator.cache[filename][row_idx][metric]['src'] = self.compute_scores(predicted_nodes_src_row, ground_truth_nodes_src_row, metric)
                    if 'tgt' not in self.cache[filename][row_idx][metric]:
                        ScoreCalculator.cache[filename][row_idx][metric]['tgt'] = self.compute_scores(predicted_nodes_tgt_row, ground_truth_nodes_tgt_row, metric)

                    all_scores_src = ScoreCalculator.cache[filename][row_idx][metric]['src']
                    all_scores_tgt = ScoreCalculator.cache[filename][row_idx][metric]['tgt']
                    
                    matrix_src = np.zeros((len(predicted_nodes_src_row), len(ground_truth_nodes_src_row)))
                    
                    for i in range(len(predicted_nodes_src_row)):
                        for j in range(len(ground_truth_nodes_src_row)):
                            matrix_src[i, j] = all_scores_src[i * len(ground_truth_nodes_src_row) + j]
                    
                    binary_matrix_src = matrix_src >= threshold
                    
                    matrix_tgt = np.zeros((len(predicted_nodes_tgt_row), len(ground_truth_nodes_tgt_row)))

                    for i in range(len(predicted_nodes_tgt_row)):
                        for j in range(len(ground_truth_nodes_tgt_row)):
                            matrix_tgt[i, j] = all_scores_tgt[i * len(ground_truth_nodes_tgt_row) + j]

                    binary_matrix_tgt = matrix_tgt >= threshold
       
                    pred_to_gt_mappings_src = self.find_node_mappings(predicted_nodes_src_row, ground_truth_nodes_src_row, binary_matrix_src)
                    pred_to_gt_mappings_tgt = self.find_node_mappings(predicted_nodes_tgt_row, ground_truth_nodes_tgt_row, binary_matrix_tgt)

                    pred_to_gt_mappings = pred_to_gt_mappings_src | pred_to_gt_mappings_tgt

                    pred_matched, matched_edges = self.find_matching_edges(predicted_edges_row, ground_truth_edges_row, pred_to_gt_mappings)

      
                    TP = len([edge for edge in matched_edges if edge in ground_truth_edges_row])
                    FP = len(predicted_edges_row) - TP 
                    FN = len(ground_truth_edges_row) - TP
                    
                    F1 = self.calculate_f1_score(TP, FP, FN)

                    row_f1_scores.append(F1)
                    detailed_scores.append({'Model': model, 'Metric': metric, 'Row': row_idx, 'F1': F1})
                detailed_scores_df = pd.DataFrame(detailed_scores)


                average_f1 = np.mean(row_f1_scores)
                overall_scores.append({'Model': model, 'Metric': metric, 'F1': average_f1})
        scores_df = pd.DataFrame(overall_scores)
        scores_df = scores_df.set_index(['Model', 'Metric']).unstack()
        scores_df.columns = ['_'.join(col).strip() for col in scores_df.columns.values]
        scores_df = scores_df.reset_index()

        return scores_df.reset_index(), detailed_scores_df


def compare_metrics_elo(elo_df, metrics_df):
    spearman_correlations = {}
    spearman_sentence_metrics = {}
    for metric in metrics_df['Metric'].unique():
        metric_df = metrics_df[metrics_df['Metric'] == metric]
        spearman_correlation = []
        spearman_sentence = {}
        for i in range(20):
            winner = elo_df[elo_df[i]==max(elo_df[i])][i].index[0]
            metric_dfi=metric_df[metric_df["Row"]==i].copy()
            elo_dfi=elo_df[i].copy()
            metric_dfi = metric_dfi[metric_dfi['Model'] != winner]
            elo_dfi = elo_dfi.drop(winner)
            
            elo_values = []
            metric_values = []
            for model in metric_dfi['Model']:
               
                elo_row_values = elo_dfi[model]
              
                metric_value = metric_dfi[metric_dfi['Model'] == model]['F1'].values[0]
                
                elo_values.append(elo_row_values)
                metric_values.append(metric_value)

            if len(set(elo_values)) <= 1 or len(set(metric_values)) <= 1:
                print(f"[Warning] Constant input detected for metric '{metric}', sentence {i}")
                print(f"elo_values: {elo_values}")
                print(f"metric_values: {metric_values}")
                S_c = 0
            else:
                S_c = spearmanr(elo_values, metric_values).correlation
                if S_c is None or np.isnan(S_c):
                    S_c = 0
         
            # S_c =spearmanr(elo_values, metric_values).correlation
            # if S_c is None or np.isnan(S_c):
            #     S_c = 0
        
            spearman_correlation.append(S_c)

            spearman_sentence[i]=S_c
        
        spearman_sentence_metrics[metric]=spearman_sentence   
        spearman_correlations[metric] = np.mean(spearman_correlation)

    closer_metric = max(spearman_correlations, key=spearman_correlations.get)

    return spearman_correlations, closer_metric, spearman_sentence_metrics


def process_hh_rlhf_dataset(dataset_name='hh_rlhf', batch_size=32):
    """Process HH-RLHF dataset and calculate metrics with GPU optimization"""
    perf_monitor.start(f"HH-RLHF metrics processing")
    print(f"🔍 Processing metrics for {dataset_name} dataset...")
    
    config = get_dataset_config(dataset_name)
    data_dir = get_data_dir(dataset_name)
    rankings_dir = get_rankings_dir(dataset_name)
    
    # Load processed data
    data_file = os.path.join(data_dir, "hh_rlhf_processed.csv")
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run the data loading step first: python hh_rlhf_loader.py")
        return False
    
    df = pd.read_csv(data_file)
    print(f"📊 Loaded {len(df)} samples")
    
    # Prepare data for metric calculation
    chosen_responses = df['chosen_response'].tolist()
    rejected_responses = df['rejected_response'].tolist()
    
    # Initialize metrics calculator with optimized batch size
    metrics_calc = Metrics(batch_size=batch_size)
    
    # Calculate all metrics in parallel batches for better performance
    print("📈 Calculating metrics with GPU optimization...")
    
    # For HH-RLHF: Extract initial human prompt and assistant responses
    # We need to separate the prompt from the assistant responses for proper metric calculation
    
    print("🧠 Calculating BLEURT scores (GPU-accelerated)...")
    
    # Extract initial human prompts and assistant responses
    human_prompts = []
    chosen_assistant_responses = []
    rejected_assistant_responses = []
    
    for chosen, rejected in zip(chosen_responses, rejected_responses):
        # Extract the initial human prompt (first "Human:" section)
        if "Human:" in chosen:
            # Get everything from the start to the first "Assistant:"
            parts = chosen.split("Assistant:")
            if len(parts) > 1:
                human_part = parts[0].replace("Human:", "").strip()
                human_prompts.append(human_part)
                
                # Extract assistant response (everything after first "Assistant:")
                assistant_part = "Assistant:".join(parts[1:]).strip()
                chosen_assistant_responses.append(assistant_part)
            else:
                # Fallback
                human_prompts.append(chosen)
                chosen_assistant_responses.append(chosen)
        else:
            # Fallback
            human_prompts.append(chosen)
            chosen_assistant_responses.append(chosen)
        
        # Extract assistant response from rejected (same logic)
        if "Human:" in rejected:
            parts = rejected.split("Assistant:")
            if len(parts) > 1:
                assistant_part = "Assistant:".join(parts[1:]).strip()
                rejected_assistant_responses.append(assistant_part)
            else:
                rejected_assistant_responses.append(rejected)
        else:
            rejected_assistant_responses.append(rejected)
    
    # Calculate metrics using assistant responses vs human prompts
    bleurt_chosen = metrics_calc.compute_bleurt(chosen_assistant_responses, human_prompts)
    bleurt_rejected = metrics_calc.compute_bleurt(rejected_assistant_responses, human_prompts)
    
    # Calculate other metrics
    print("Calculating BLEU scores...")
    bleu_chosen = metrics_calc.compute_bleu(chosen_assistant_responses, human_prompts)
    bleu_rejected = metrics_calc.compute_bleu(rejected_assistant_responses, human_prompts)
    
    print("Calculating METEOR scores...")
    meteor_chosen = metrics_calc.compute_meteor(chosen_assistant_responses, human_prompts)
    meteor_rejected = metrics_calc.compute_meteor(rejected_assistant_responses, human_prompts)
    
    print("Calculating ROUGE scores...")
    rouge_chosen = metrics_calc.compute_rouge(chosen_assistant_responses, human_prompts)
    rouge_rejected = metrics_calc.compute_rouge(rejected_assistant_responses, human_prompts)
    
    # Calculate verbatim scores (simple string comparison)
    print("Calculating verbatim scores...")
    verbatim_chosen = [1.0 if chosen.strip() == prompt.strip() else 0.0 
                      for chosen, prompt in zip(chosen_assistant_responses, human_prompts)]
    verbatim_rejected = [1.0 if rejected.strip() == prompt.strip() else 0.0 
                       for rejected, prompt in zip(rejected_assistant_responses, human_prompts)]
    
    # Build metrics data efficiently
    metrics_data = []
    for i in range(len(df)):
        # Chosen response metrics (winner of pairwise comparison)
        metrics_data.append({
            'Row': i,
            'Model': 'chosen',
            'bleu': bleu_chosen[i],
            'meteor': meteor_chosen[i],
            'rouge': rouge_chosen[i],
            'verbatim': verbatim_chosen[i],
            'bleurt': bleurt_chosen[i],
            'text': chosen_assistant_responses[i][:100] + "..."
        })
        
        # Rejected response metrics (loser of pairwise comparison)
        metrics_data.append({
            'Row': i,
            'Model': 'rejected',
            'bleu': bleu_rejected[i],
            'meteor': meteor_rejected[i],
            'rouge': rouge_rejected[i],
            'verbatim': verbatim_rejected[i],
            'bleurt': bleurt_rejected[i],
            'text': rejected_assistant_responses[i][:100] + "..."
        })
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create rankings directory
    os.makedirs(rankings_dir, exist_ok=True)
    
    # Create detailed scores file
    print("Creating detailed scores file...")
    detailed_scores = []
    for _, row in metrics_df.iterrows():
        for metric in ['bleu', 'bleurt', 'meteor', 'rouge', 'verbatim']:
            detailed_scores.append({
                'Row': row['Row'],
                'Model': row['Model'],
                'Metric': metric,
                'F1': row[metric]
            })
    
    detailed_df = pd.DataFrame(detailed_scores)
    detailed_file = os.path.join(data_dir, "detailed_scores.csv")
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Saved detailed_scores.csv")
    
    perf_monitor.end(f"HH-RLHF metrics processing")
    perf_monitor.print_summary()
    
    print("✅ HH-RLHF metrics calculation complete!")
    return True

def process_summarize_feedback_dataset(dataset_name='summarize_feedback', batch_size=32):
    """Process summarize-from-feedback dataset and calculate metrics with GPU optimization"""
    perf_monitor.start(f"Summarize-feedback metrics processing")
    print(f"🔍 Processing metrics for {dataset_name} dataset...")
    
    config = get_dataset_config(dataset_name)
    data_dir = get_data_dir(dataset_name)
    rankings_dir = get_rankings_dir(dataset_name)
    
    # Load processed data
    data_file = os.path.join(data_dir, "summarize_feedback_processed.csv")
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run the data loading step first: python summarize_feedback_loader.py")
        return False
    
    df = pd.read_csv(data_file)
    print(f"📊 Loaded {len(df)} samples")
    
    # Prepare data for metric calculation
    # Reference: Original post text (what the summaries are trying to summarize)
    # Candidates: Winner and loser summaries (to be evaluated against the reference)
    winner_texts = df['winner_text'].tolist()
    loser_texts = df['loser_text'].tolist()
    post_texts = df['post_text'].tolist()  # This is our reference text
    
    # Initialize metrics calculator with optimized batch size
    metrics_calc = Metrics(batch_size=batch_size)
    
    # Calculate all metrics in parallel batches for better performance
    print("📈 Calculating metrics with GPU optimization...")
    
    # For summarize-feedback: Use original post text as reference, calculate metrics for both summaries
    print("🧠 Calculating BLEURT scores (GPU-accelerated)...")
    print("📝 Reference: Original post text")
    print("📊 Candidates: Winner and loser summaries")
    
    # Calculate metrics using summaries vs post text (reference)
    bleurt_winner = metrics_calc.compute_bleurt(winner_texts, post_texts)
    bleurt_loser = metrics_calc.compute_bleurt(loser_texts, post_texts)
    
    # Calculate other metrics
    print("Calculating BLEU scores...")
    bleu_winner = metrics_calc.compute_bleu(winner_texts, post_texts)
    bleu_loser = metrics_calc.compute_bleu(loser_texts, post_texts)
    
    print("Calculating METEOR scores...")
    meteor_winner = metrics_calc.compute_meteor(winner_texts, post_texts)
    meteor_loser = metrics_calc.compute_meteor(loser_texts, post_texts)
    
    print("Calculating ROUGE scores...")
    rouge_winner = metrics_calc.compute_rouge(winner_texts, post_texts)
    rouge_loser = metrics_calc.compute_rouge(loser_texts, post_texts)
    
    # Calculate verbatim scores (simple string comparison)
    print("Calculating verbatim scores...")
    verbatim_winner = [1.0 if winner.strip() == post.strip() else 0.0 
                      for winner, post in zip(winner_texts, post_texts)]
    verbatim_loser = [1.0 if loser.strip() == post.strip() else 0.0 
                     for loser, post in zip(loser_texts, post_texts)]
    
    # Build metrics data efficiently
    metrics_data = []
    for i in range(len(df)):
        # Winner summary metrics (winner of pairwise comparison)
        metrics_data.append({
            'Row': i,
            'Model': 'winner',
            'bleu': bleu_winner[i],
            'meteor': meteor_winner[i],
            'rouge': rouge_winner[i],
            'verbatim': verbatim_winner[i],
            'bleurt': bleurt_winner[i],
            'text': winner_texts[i][:100] + "...",
            'policy': df.iloc[i]['winner_policy']
        })
        
        # Loser summary metrics (loser of pairwise comparison)
        metrics_data.append({
            'Row': i,
            'Model': 'loser',
            'bleu': bleu_loser[i],
            'meteor': meteor_loser[i],
            'rouge': rouge_loser[i],
            'verbatim': verbatim_loser[i],
            'bleurt': bleurt_loser[i],
            'text': loser_texts[i][:100] + "...",
            'policy': df.iloc[i]['loser_policy']
        })
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create rankings directory
    os.makedirs(rankings_dir, exist_ok=True)
    
    # Create detailed scores file
    print("💾 Creating detailed scores file...")
    detailed_scores = []
    for _, row in metrics_df.iterrows():
        for metric in ['bleu', 'bleurt', 'meteor', 'rouge', 'verbatim']:
            detailed_scores.append({
                'Row': row['Row'],
                'Model': row['Model'],
                'Metric': metric,
                'F1': row[metric]
            })
    
    detailed_df = pd.DataFrame(detailed_scores)
    detailed_file = os.path.join(data_dir, "detailed_scores.csv")
    detailed_df.to_csv(detailed_file, index=False)
    print(f"💾 Saved detailed_scores.csv")
    
    # Create individual metric ranking files for reg_test.py
    print("📊 Creating metric ranking tables...")
    for metric in config['metrics']:
        metric_df = detailed_df[detailed_df['Metric'] == metric]
        if len(metric_df) == 0:
            print(f"  ⚠️ No data found for metric: {metric}")
            continue
        
        # Pivot to create the format expected by reg_test.py
        # Rows are sample indices, columns are winner/loser
        pivot = metric_df.pivot(index='Row', columns='Model', values='F1')
        
        # Save to rankings directory
        metric_file = os.path.join(rankings_dir, f"{metric}_values.csv")
        pivot.to_csv(metric_file)
        print(f"  ✅ Created {metric}_values.csv")
    
    perf_monitor.end(f"Summarize-feedback metrics processing")
    perf_monitor.print_summary()
    
    print("✅ Summarize-feedback metrics calculation complete!")
    return True

def process_causal_relations_dataset(dataset_name='causal_relations'):
    """Process causal relations dataset (original logic)"""
    print(f"🔍 Processing metrics for {dataset_name} dataset...")
    
    config = get_dataset_config(dataset_name)
    processed_data_dir = get_processed_data_dir(dataset_name)
    data_dir = get_data_dir(dataset_name)
    
    # Use original logic for causal relations
    filenames = {name: os.path.join(processed_data_dir, f"{name}_elo_full.json") 
                for name in config['annotator_mapping'].values()}
    
    elo_file = os.path.join(data_dir, "final_elo_rankings.csv")
    elo = pd.read_csv(elo_file)
    elo_T = elo.drop(columns=["sentence_start", "winner"]).transpose()
    
    # Use configuration from config.py
    threshold_sets = THRESHOLD_SETS
    best_thresholds = {}
    
    for metric in tqdm(config['metrics'], desc="Tuning metrics"):
        max_spearman = float('-inf')
        best_threshold = None
        for thresholds in tqdm(threshold_sets, desc=f"Thresholds for {metric}", leave=False):
            scoring = ScoreCalculator(filenames, [metric], thresholds)
            scores_df, detail = scoring.calculate_scores()
            spearman_correlation, _, _ = compare_metrics_elo(elo_T, detail)
            if spearman_correlation[metric] > max_spearman:
                max_spearman = spearman_correlation[metric]
                best_threshold = thresholds[metric]
        best_thresholds[metric] = best_threshold
    
    print("✅ Causal relations metrics calculation complete!")
    return True

def main():
    """Main function to handle all datasets with GPU optimization"""
    parser = argparse.ArgumentParser(description="Calculate metrics for datasets with GPU optimization")
    parser.add_argument("--dataset", choices=["causal_relations", "hh_rlhf", "summarize_feedback"], 
                       default="causal_relations", help="Dataset to process")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for GPU processing (default: 32)")
    parser.add_argument("--gpu", action="store_true", 
                       help="Enable GPU acceleration (auto-detected)")
    
    args = parser.parse_args()
    
    print("🚀 Metric Calculation with GPU Optimization")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU available: {len(gpus) > 0}")
    if gpus:
        print(f"GPU devices: {[gpu.name for gpu in gpus]}")
    print("=" * 50)
    
    if args.dataset == 'hh_rlhf':
        success = process_hh_rlhf_dataset(args.dataset, batch_size=args.batch_size)
    elif args.dataset == 'summarize_feedback':
        success = process_summarize_feedback_dataset(args.dataset, batch_size=args.batch_size)
    else:
        success = process_causal_relations_dataset(args.dataset)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
