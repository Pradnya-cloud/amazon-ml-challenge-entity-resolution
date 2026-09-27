"""
Data loader module for Business Entity Resolution.

Handles loading of large TSV files efficiently, with proper encoding and
tab-separated parsing. Also provides train/validation split utilities
that preserve the Macro F0.5 evaluation semantics.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))


def _resolve_dataset_dir():
    """Find the dataset directory."""
    candidates = [
        os.path.join(BASE_DIR, 'dataset'),
        os.path.join(os.path.dirname(BASE_DIR), '6ab10eb3b23ba_student_resource', 'student_resource', 'dataset'),
        '/kaggle/input/dataset',
        '/data/dataset',
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, 'train', 'train_source1.tsv')):
            return c
    raise FileNotFoundError(f"Dataset directory not found. Checked: {candidates}")


DATASET_DIR = os.environ.get('DATASET_DIR', _resolve_dataset_dir())


def load_source1(split: str = 'train') -> pd.DataFrame:
    """Load Source 1 data from TSV file."""
    path = os.path.join(DATASET_DIR, 'train' if split == 'train' else 'test',
                        f'{split}_source1.tsv')
    return _load_tsv(path)


def load_source2(split: str = 'train') -> pd.DataFrame:
    """Load Source 2 data from TSV file."""
    path = os.path.join(DATASET_DIR, 'train' if split == 'train' else 'test',
                        f'{split}_source2.tsv')
    return _load_tsv(path)


def load_source3(split: str = 'train') -> pd.DataFrame:
    """Load Source 3 data from TSV file."""
    path = os.path.join(DATASET_DIR, 'train' if split == 'train' else 'test',
                        f'{split}_source3.tsv')
    return _load_tsv(path)


def _load_tsv(path: str) -> pd.DataFrame:
    """Load a TSV file with proper encoding and tab separator."""
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_values=[''], encoding='utf-8')
    # Fill NaN with empty string for consistency
    for col in ['business_name', 'business_address', 'country']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    return df


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth from TSV file."""
    path = os.path.join(DATASET_DIR, 'train', 'train_ground_truth.tsv')
    gt = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_values=[''], encoding='utf-8')
    return gt


def ground_truth_to_dict(gt: pd.DataFrame) -> Dict[str, List[str]]:
    """Convert ground truth DataFrame to dict mapping S1 ID -> list of matched IDs."""
    s1_ids = gt['source1_entity_id'].tolist()
    match_lists = gt['matched_entity_ids'].tolist()
    result = {}
    for i in range(len(s1_ids)):
        matches = match_lists[i]
        if pd.isna(matches) or matches == '' or matches == 'nan':
            result[s1_ids[i]] = []
        else:
            result[s1_ids[i]] = matches.split(',')
    return result


def load_all_data(split: str = 'train') -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load all three sources for a given split."""
    s1 = load_source1(split)
    s2 = load_source2(split)
    s3 = load_source3(split)
    return s1, s2, s3


def create_train_val_split(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    val_frac: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, List[str]],
           pd.DataFrame, Dict[str, List[str]]]:
    """
    Split S1 entities into train and validation sets.

    Returns:
        train_s1, train_gt_dict, val_s1, val_gt_dict
    """
    rng = np.random.RandomState(random_state)
    n = len(s1)
    indices = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_s1 = s1.iloc[train_idx].reset_index(drop=True)
    val_s1 = s1.iloc[val_idx].reset_index(drop=True)

    gt_dict = ground_truth_to_dict(gt)
    train_gt = {k: v for k, v in gt_dict.items() if k in set(train_s1['entity_id'])}
    val_gt = {k: v for k, v in gt_dict.items() if k in set(val_s1['entity_id'])}

    return train_s1, train_gt, val_s1, val_gt


def sample_training_entities(
    s1: pd.DataFrame,
    gt: pd.DataFrame,
    n_samples: int = 100000,
    min_matches: int = 0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample a manageable subset of S1 entities for training.

    Ensures a mix of singletons and matched entities.
    """
    gt_dict = ground_truth_to_dict(gt)
    gt_df = pd.DataFrame([
        {'source1_entity_id': k, 'n_matches': len(v)} for k, v in gt_dict.items()
    ])

    # Merge match counts
    s1_merged = s1.merge(gt_df, left_on='entity_id', right_on='source1_entity_id', how='left')
    s1_merged['n_matches'] = s1_merged['n_matches'].fillna(0).astype(int)

    # Stratified sampling: ensure both singletons and multi-match entities
    matched = s1_merged[s1_merged['n_matches'] > 0]
    unmatched = s1_merged[s1_merged['n_matches'] == 0]

    n_matched = int(n_samples * 0.6)
    n_unmatched = n_samples - n_matched

    n_matched = min(n_matched, len(matched))
    n_unmatched = min(n_unmatched, len(unmatched))

    sampled_matched = matched.sample(n=n_matched, random_state=random_state)
    sampled_unmatched = unmatched.sample(n=n_unmatched, random_state=random_state)

    sampled_s1 = pd.concat([sampled_matched, sampled_unmatched], ignore_index=True)
    sampled_gt = gt[gt['source1_entity_id'].isin(sampled_s1['entity_id'])].reset_index(drop=True)

    return sampled_s1, sampled_gt


def get_data_stats(df: pd.DataFrame) -> Dict:
    """Get basic statistics about a source dataframe."""
    stats = {
        'n_records': len(df),
        'n_unique_ids': df['entity_id'].nunique(),
        'missing_name': df['business_name'].isna().sum() + (df['business_name'] == '').sum(),
        'missing_address': df['business_address'].isna().sum() + (df['business_address'] == '').sum(),
        'missing_country': df['country'].isna().sum() + (df['country'] == '').sum(),
        'country_values': dict(df['country'].value_counts()),
        'duplicate_ids': df['entity_id'].duplicated().sum(),
    }
    return stats
