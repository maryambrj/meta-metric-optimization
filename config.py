"""
Configuration file for Meta-Metric Optimization
Supports multiple datasets: causal_relations and hh_rlhf
"""

import os

# Project directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
BLEURT_DIR = os.path.join(PROJECT_ROOT, "bleurt")
CORE_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "core_scripts")

# Dataset configurations
DATASETS = {
    'causal_relations': {
        'name': 'Causal Relations Extraction',
        'description': 'Human and LLM annotations for causal relation extraction',
        'base_dir': os.path.join(DATASETS_DIR, "causal_relations"),
        'data_dir': os.path.join(DATASETS_DIR, "causal_relations", "data"),
        'annotations_dir': os.path.join(DATASETS_DIR, "causal_relations", "annotations"),
        'processed_data_dir': os.path.join(DATASETS_DIR, "causal_relations", "processed_data"),
        'rankings_dir': os.path.join(DATASETS_DIR, "causal_relations", "rankings"),
        'num_samples': 20,
        'num_annotators': 10,
        'metrics': ['bleu', 'bleurt', 'meteor', 'rouge', 'verbatim'],
        'annotator_mapping': {
            'Aadarsh': 'Human1',
            'Ashlin': 'Human2', 
            'Kuldeep': 'Human3',
            'Maryam': 'Human4',
            'Nate': 'Human5',
            'Riley': 'Human6',
            'Spencer': 'Human7',
            'llama2': 'llama2',
            'llama3': 'llama3',
            'mistral': 'mistral'
        },
        'ordered_columns': [
            'Human1', 'Human2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7',
            'llama2', 'llama3', 'mistral'
        ]
    },
    'hh_rlhf': {
        'name': 'Anthropic HH-RLHF',
        'description': 'Human preference dataset with chosen/rejected responses',
        'base_dir': os.path.join(DATASETS_DIR, "hh_rlhf"),
        'data_dir': os.path.join(DATASETS_DIR, "hh_rlhf", "data"),
        'annotations_dir': os.path.join(DATASETS_DIR, "hh_rlhf", "annotations"),
        'processed_data_dir': os.path.join(DATASETS_DIR, "hh_rlhf", "processed_data"),
        'rankings_dir': os.path.join(DATASETS_DIR, "hh_rlhf", "rankings"),
        'num_samples': None,  # Will be determined from dataset
        'num_annotators': 2,  # chosen vs rejected
        'metrics': ['bleu', 'bleurt', 'meteor', 'rouge', 'verbatim'],
        'annotator_mapping': {
            'chosen': 'chosen',
            'rejected': 'rejected'
        },
        'ordered_columns': ['chosen', 'rejected']
    }
}

# Default dataset (can be changed)
DEFAULT_DATASET = 'causal_relations'

# Helper functions to get dataset-specific paths
def get_dataset_config(dataset_name=None):
    """Get configuration for a specific dataset"""
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
    return DATASETS.get(dataset_name, DATASETS[DEFAULT_DATASET])

def get_data_dir(dataset_name=None):
    """Get data directory for a specific dataset"""
    return get_dataset_config(dataset_name)['data_dir']

def get_annotations_dir(dataset_name=None):
    """Get annotations directory for a specific dataset"""
    return get_dataset_config(dataset_name)['annotations_dir']

def get_processed_data_dir(dataset_name=None):
    """Get processed data directory for a specific dataset"""
    return get_dataset_config(dataset_name)['processed_data_dir']

def get_rankings_dir(dataset_name=None):
    """Get rankings directory for a specific dataset"""
    return get_dataset_config(dataset_name)['rankings_dir']

# Backward compatibility - use default dataset
DATA_DIR = get_data_dir()
ANNOTATIONS_DIR = get_annotations_dir()
PROCESSED_DATA_DIR = get_processed_data_dir()
RANKINGS_DIR = get_rankings_dir()

# Data files for causal_relations dataset (backward compatibility)
WINNER_ANNOTATIONS_FILE = os.path.join(DATA_DIR, "winner_annotations.csv")
FINAL_ELO_RANKINGS_FILE = os.path.join(DATA_DIR, "final_elo_rankings.csv")
DETAILED_SCORES_FILE = os.path.join(DATA_DIR, "detailed_scores.csv")

# BLEURT checkpoint
BLEURT_CHECKPOINT = os.path.join(BLEURT_DIR, "BLEURT-20")

# Annotator mapping
ANNOTATOR_MAPPING = {
    'Aadarsh': 'Human1',
    'Ashlin': 'Human2', 
    'Kuldeep': 'Human3',
    'Maryam': 'Human4',
    'Nate': 'Human5',
    'Riley': 'Human6',
    'Spencer': 'Human7',
    'llama2': 'llama2',
    'llama3': 'llama3',
    'mistral': 'mistral'
}

# Ordered columns for consistency
ORDERED_COLUMNS = [
    'Human1', 'Human2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7',
    'llama2', 'llama3', 'mistral'
]

# Metrics configuration
METRICS = ['bleu', 'bleurt', 'meteor', 'rouge', 'verbatim']

# Threshold sets for metric optimization
THRESHOLD_SETS = [
    {'bleurt': -.05, 'bleu': .005, 'meteor': .005, 'rouge': .1, 'verbatim': 0.5},
    {'bleurt': -.1, 'bleu': 0.01, 'meteor': .01, 'rouge': 0.1, 'verbatim': 0.5},
    {'bleurt': -.1, 'bleu': 0.02, 'meteor': .1, 'rouge': 0.1, 'verbatim': 0.5},
    {'bleurt': -.3, 'bleu': 0.07, 'meteor': 0.25, 'rouge': 0.3, 'verbatim': 0.5},
    {'bleurt': -.7, 'bleu': 0.07, 'meteor': 0.4, 'rouge': 0.5, 'verbatim': 0.5},
    {'bleurt': -1, 'bleu': 0.2, 'meteor': 0.5, 'rouge': 0.75, 'verbatim': 0.5},
    {'bleurt': -1.1, 'bleu': 0.2, 'meteor': 0.5, 'rouge': 0.75, 'verbatim': 0.5}
]

# Model names for processed files
MODEL_NAMES = [
    "Human1", "Human2", "Human3", "Human4", "Human5", "Human6", "Human7",
    "llama2", "llama3", "mistral"
]

# Number of samples
NUM_SAMPLES = 20
NUM_ANNOTATORS = 10

# TensorFlow settings
TF_CPP_MIN_LOG_LEVEL = '2' 