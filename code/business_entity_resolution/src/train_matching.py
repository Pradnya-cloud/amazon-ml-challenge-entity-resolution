"""
Training module for Business Entity Resolution.

Samples training data, generates candidate pairs, extracts features,
trains a LightGBM model, and optimizes the probability threshold for
Macro F0.5.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    load_source1, load_source2, load_source3, load_ground_truth,
    ground_truth_to_dict, sample_training_entities, create_train_val_split,
    DATASET_DIR
)
from preprocess import normalize_name_compact, normalize_name_compact_series, normalize_country, normalize_country_series, normalize_address
from blocking import (
    fit_tfidf_vectorizer, generate_all_candidates,
    candidates_to_candidate_pairs, evaluate_blocking_recall,
    COSINE_THRESHOLD, TOP_K_CANDIDATES
)
from feature_extraction import (
    extract_features_pair, FEATURE_COLUMNS, extract_features_batch,
    set_country_pair_freq
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
SCRATCH_DIR = os.path.join(BASE_DIR, 'scratch')
MODELS_DIR = os.path.join(SCRATCH_DIR, 'models')

# Row limit when loading S1/GT/S2/S3 for training (to avoid processing all 2M+ rows)
TRAIN_LOAD_LIMIT = 50000
NEG_PER_COUNTRY = 30000

US_COUNTRY_VALUES = {'us', 'usa', 'united states', 'united states of america', 'u.s.', 'u.s.a'}
INDIA_COUNTRY_VALUES = {'india', 'in', 'republic of india', 'bharat'}


def _classify_country_fast(country_series):
    """Fast vectorized country classification to US/India/France/Other."""
    lower = country_series.astype(str).str.lower().str.strip()
    result = np.where(lower.isin(US_COUNTRY_VALUES), 'US',
             np.where(lower.isin(INDIA_COUNTRY_VALUES), 'India',
             np.where(lower.str.startswith('fr'), 'France', 'Other')))
    return pd.Series(result, index=country_series.index)


def load_training_data_subset(
    n_s1_rows: int = TRAIN_LOAD_LIMIT,
    gt_dict: Dict[str, List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load a subset of training data sufficient for model training.

    Loads first N rows of each source file to get a manageable dataset
    with proper S1/GT ID alignment and sufficient positive/negative examples.
    """
    from data_loader import DATASET_DIR

    s1_path = os.path.join(DATASET_DIR, 'train', 'train_source1.tsv')
    s2_path = os.path.join(DATASET_DIR, 'train', 'train_source2.tsv')
    s3_path = os.path.join(DATASET_DIR, 'train', 'train_source3.tsv')
    gt_path = os.path.join(DATASET_DIR, 'train', 'train_ground_truth.tsv')

    print(f"  Loading {n_s1_rows} rows of S1...")
    s1 = pd.read_csv(s1_path, sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False, nrows=n_s1_rows)

    print(f"  Loading {n_s1_rows} rows of GT...")
    gt = pd.read_csv(gt_path, sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False, nrows=n_s1_rows)
    gt = gt.drop_duplicates(subset='source1_entity_id', keep='first')
    gt_dict = ground_truth_to_dict(gt)

    # Find aligned S1 IDs (in both S1 file and GT)
    s1_ids = set(s1['entity_id'])
    aligned_ids = s1_ids & set(gt['source1_entity_id'])
    print(f"  Aligned S1 IDs: {len(aligned_ids)}", flush=True)

    # Filter S1 to only aligned entities
    s1 = s1[s1['entity_id'].isin(aligned_ids)].reset_index(drop=True)
    print(f"  S1 filtered: {len(s1)}", flush=True)

    # Filter GT to only aligned IDs and build dict (critical for perf)
    gt = gt[gt['source1_entity_id'].isin(aligned_ids)].reset_index(drop=True)
    gt_dict = ground_truth_to_dict(gt)
    print(f"  GT dict (filtered): {len(gt_dict)} entries", flush=True)

    # Collect all GT match IDs for S2 and S3
    gt_s2_ids = set()
    gt_s3_ids = set()
    for s1_id, matches in gt_dict.items():
        for mid in matches:
            if mid.startswith('S2-'):
                gt_s2_ids.add(mid)
            elif mid.startswith('S3-'):
                gt_s3_ids.add(mid)
    print(f"  GT match IDs: S2={len(gt_s2_ids)}, S3={len(gt_s3_ids)}", flush=True)

    # Load S2/S3 - first pass to get matching rows, then sample negatives
    def load_source_with_gt(path, gt_ids, source_name):
        # First, scan with entity_id + country only to find matches + classify negatives
        df = pd.read_csv(path, sep='\t', encoding='utf-8', dtype=str,
                         keep_default_na=False, nrows=n_s1_rows)
        return df

    s2 = load_source_with_gt(s2_path, gt_s2_ids, 'S2')
    s3 = load_source_with_gt(s3_path, gt_s3_ids, 'S3')

    print(f"  S2 loaded: {len(s2)}, S3 loaded: {len(s3)}")
    return s1, s2, s3, gt_dict


def run_training_pipeline(
    n_train_samples: int = 3000,
    n_val_samples: int = 2000,
    cosine_threshold: float = 0.05,
    top_k: int = 50,
):
    """
    Full training pipeline: load data, generate features, train model, optimize threshold.
    """
    print("=" * 60)
    print("Loading training data (subset)...")
    print("=" * 60)

    s1, s2, s3, gt_dict = load_training_data_subset(n_s1_rows=TRAIN_LOAD_LIMIT)

    print(f"S1: {len(s1)} entities, S2: {len(s2)}, S3: {len(s3)}")
    print(f"Ground truth: {len(gt_dict)} S1 entities")

    # Random split
    random_state = 42
    rng = np.random.RandomState(random_state)

    all_s1_ids = list(gt_dict.keys())
    rng.shuffle(all_s1_ids)

    n_train = min(n_train_samples, int(len(all_s1_ids) * 0.8))
    n_val = min(n_val_samples, int(len(all_s1_ids) * 0.2))

    train_s1_ids = all_s1_ids[:n_train]
    val_s1_ids = all_s1_ids[n_train:n_train + n_val]

    print(f"Training S1 entities: {len(train_s1_ids)}")
    print(f"Validation S1 entities: {len(val_s1_ids)}")

    train_s1 = s1[s1['entity_id'].isin(train_s1_ids)].reset_index(drop=True)
    print(f"  train_s1 filtered: {len(train_s1)}", flush=True)
    val_s1 = s1[s1['entity_id'].isin(val_s1_ids)].reset_index(drop=True)
    print(f"  val_s1 filtered: {len(val_s1)}", flush=True)

    # Filter GT dict using pandas instead of dict comprehension (avoid iterating 113K dict)
    train_s1_id_set = set(train_s1['entity_id'])
    val_s1_id_set = set(val_s1['entity_id'])
    train_gt = {k: gt_dict[k] for k in train_s1_id_set if k in gt_dict}
    val_gt = {k: gt_dict[k] for k in val_s1_id_set if k in gt_dict}
    print(f"  train_gt built: {len(train_gt)}", flush=True)
    print(f"  val_gt built: {len(val_gt)}", flush=True)

    print(f"Train S1: {len(train_s1)}, GT: {len(train_gt)}")
    print(f"Val S1: {len(val_s1)}, GT: {len(val_gt)}")

    # Fit TF-IDF vectorizer on training data names
    print("\nFitting TF-IDF vectorizer...", flush=True)
    all_names = (
        normalize_name_compact_series(train_s1['business_name']).tolist() +
        normalize_name_compact_series(s2['business_name']).tolist() +
        normalize_name_compact_series(s3['business_name']).tolist()
    )
    vectorizer = fit_tfidf_vectorizer(all_names)
    print(f"  TF-IDF vocab size: {len(vectorizer.vocabulary_)}", flush=True)

    # Compute country-pair match frequency from ground truth
    print("\nComputing country-pair frequency...", flush=True)
    s2_country_lookup = dict(zip(s2['entity_id'], _classify_country_fast(s2['country'])))
    s3_country_lookup = dict(zip(s3['entity_id'], _classify_country_fast(s3['country'])))
    s1_country_lookup = dict(zip(train_s1['entity_id'], _classify_country_fast(train_s1['country'])))
    print(f"  Country lookups built", flush=True)

    # Count matches per country pair
    pair_match_counts = {}
    for s1_id, matches in train_gt.items():
        s1_cn = s1_country_lookup.get(s1_id, '')
        if s1_cn == '':
            continue
        for mid in matches:
            if mid.startswith('S2-'):
                mid_cn = s2_country_lookup.get(mid, '')
            elif mid.startswith('S3-'):
                mid_cn = s3_country_lookup.get(mid, '')
            else:
                continue
            key = (s1_cn, mid_cn or 'Unknown')
            pair_match_counts[key] = pair_match_counts.get(key, 0) + 1

    # Count total entities per country (for frequency calculation)
    s1_country_counts = normalize_country_series(train_s1['country']).value_counts().to_dict()
    s2_country_counts = normalize_country_series(s2['country']).value_counts().to_dict()
    s3_country_counts = normalize_country_series(s3['country']).value_counts().to_dict()

    # Frequency = matches / possible_pairs for each country pair
    freq_table = {}
    for (s1_cn, s2_cn), n_matches in pair_match_counts.items():
        if s2_cn == 'Unknown':
            freq_table[(s1_cn, s2_cn)] = n_matches
        else:
            s2_count = s2_country_counts.get(s2_cn, 1) + s3_country_counts.get(s2_cn, 1)
            s1_count = s1_country_counts.get(s1_cn, 1)
            if s1_count > 0 and s2_count > 0:
                freq_table[(s1_cn, s2_cn)] = min(n_matches / (s1_count * s2_count), 1.0)
            else:
                freq_table[(s1_cn, s2_cn)] = n_matches / 1000.0

    print(f"  Country-pair freq entries: {len(freq_table)}")
    set_country_pair_freq(freq_table)

    # Generate candidates and features for training
    print("\nPreparing training data...", flush=True)
    X_train_df, y_train = prepare_training_data(
        train_s1, s2, s3, train_gt,
        vectorizer=vectorizer,
        cosine_threshold=cosine_threshold,
        top_k=top_k,
    )
    print(f"Train features: {X_train_df.shape}, labels: {y_train.shape}")

    # Generate candidates and features for validation
    print("\nPreparing validation data...", flush=True)
    X_val_df, y_val = prepare_training_data(
        val_s1, s2, s3, val_gt,
        vectorizer=vectorizer,
        cosine_threshold=cosine_threshold,
        top_k=top_k,
    )
    print(f"Val features: {X_val_df.shape}, labels: {y_val.shape}")

    # Encode source column as float
    X_train_df['source'] = X_train_df['source'].apply(lambda x: 0.0 if x == 'S2' else 1.0)
    X_val_df['source'] = X_val_df['source'].apply(lambda x: 0.0 if x == 'S2' else 1.0)

    # Convert all feature columns to float to avoid LightGBM categorical mismatch
    for col in FEATURE_COLUMNS + ['blocking_score', 'source']:
        X_train_df[col] = X_train_df[col].astype(float)
        X_val_df[col] = X_val_df[col].astype(float)

    # Train model
    print("\nTraining LightGBM model...", flush=True)
    feature_cols = FEATURE_COLUMNS + ['blocking_score', 'source']
    model, best_threshold, best_stats = train_model(
        X_train_df, y_train, X_val_df, y_val,
        X_val_df['s1_id'], val_gt
    )

    # Save artifacts to scratch/models/
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, 'lgbm_model.pkl')
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    threshold_path = os.path.join(MODELS_DIR, 'threshold.txt')
    features_path = os.path.join(MODELS_DIR, 'feature_columns.txt')
    freq_path = os.path.join(MODELS_DIR, 'country_pair_freq.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(threshold_path, 'w') as f:
        f.write(str(best_threshold))
    with open(features_path, 'w') as f:
        for col in feature_cols:
            f.write(col + '\n')
    with open(freq_path, 'wb') as f:
        pickle.dump(freq_table, f)

    print(f"\nArtifacts saved to {MODELS_DIR}/")
    print(f"Model: {model_path}")
    print(f"Vectorizer: {vectorizer_path}")
    print(f"Threshold: {threshold_path} = {best_threshold:.4f}")
    print(f"Features: {features_path}")
    print(f"Country-pair freq: {freq_path}")

    return model, best_threshold, vectorizer, best_stats


def compute_f05_precision_recall(
    true_matches: List[str], predicted: List[str]
) -> Tuple[float, float, float]:
    """
    Compute F0.5, Precision, Recall for a single S1 entity.

    For singletons (true_matches empty):
      - If predicted is empty: F0.5 = 1.0 (true negative)
      - If predicted is non-empty: F0.5 = 0.0 (false positive)

    Otherwise:
      F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    tp = len(set(true_matches) & set(predicted))
    fp = len(set(predicted) - set(true_matches))
    fn = len(set(true_matches) - set(predicted))

    if len(true_matches) == 0:
        # Singleton
        if len(predicted) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    else:
        # Has true matches
        if len(predicted) == 0:
            return 0.0, 0.0, 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision > 0 and recall > 0:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            else:
                f05 = 0.0
            return f05, precision, recall


def compute_macro_f05(
    gt_dict: Dict[str, List[str]],
    predictions: Dict[str, List[str]]
) -> dict:
    """
    Compute Macro F0.5 score across all S1 entities.

    Args:
        gt_dict: S1 entity_id -> list of true matched IDs
        predictions: S1 entity_id -> list of predicted matched IDs

    Returns: dict with macro_f05, macro_precision, macro_recall, etc.
    """
    f05_scores = []
    precisions = []
    recalls = []
    singleton_f05 = []
    matched_f05 = []

    for s1_id, true_matches in gt_dict.items():
        predicted = predictions.get(s1_id, [])
        f05, p, r = compute_f05_precision_recall(true_matches, predicted)
        f05_scores.append(f05)
        precisions.append(p)
        recalls.append(r)

        if len(true_matches) == 0:
            singleton_f05.append(f05)
        else:
            matched_f05.append(f05)

    results = {
        'macro_f05': np.mean(f05_scores) if f05_scores else 0.0,
        'macro_precision': np.mean(precisions) if precisions else 0.0,
        'macro_recall': np.mean(recalls) if recalls else 0.0,
        'n_entities': len(f05_scores),
        'n_singletons': len(singleton_f05),
        'singleton_f05': np.mean(singleton_f05) if singleton_f05 else 0.0,
        'n_matched': len(matched_f05),
        'matched_f05': np.mean(matched_f05) if matched_f05 else 0.0,
        'n_correct_singletons': sum(1 for f in singleton_f05 if f >= 1.0),
        'n_false_positive_singletons': sum(1 for f in singleton_f05 if f < 1.0),
    }
    return results


def prepare_training_data(
    s1_sample: pd.DataFrame,
    s2_full: pd.DataFrame,
    s3_full: pd.DataFrame,
    gt_dict: Dict[str, List[str]],
    vectorizer: TfidfVectorizer = None,
    cosine_threshold: float = 0.1,
    top_k: int = 200,
    scratch_dir: str = SCRATCH_DIR,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate candidate pairs, extract features, and create labels for training.
    """
    os.makedirs(scratch_dir, exist_ok=True)

    # Generate candidates using blocking
    candidates = generate_all_candidates(
        s1_sample, s2_full, s3_full,
        vectorizer=vectorizer,
        cache_dir=scratch_dir,
        gt_dict=gt_dict,
        cosine_threshold=cosine_threshold,
        top_k=top_k,
    )

    # Evaluate blocking
    blocking_stats = evaluate_blocking_recall(candidates, gt_dict)
    print(f"Blocking stats: {blocking_stats}")

    # Build feature rows
    feature_rows = []
    labels = []

    # Create lookup DataFrames for efficient access
    s2_lookup = s2_full.set_index('entity_id')
    s3_lookup = s3_full.set_index('entity_id')

    for s1_id, cand_list in candidates.items():
        if len(cand_list) == 0:
            continue

        # Get S1 entity info
        s1_rows = s1_sample[s1_sample['entity_id'] == s1_id]
        if len(s1_rows) == 0:
            continue
        s1_row = s1_rows.iloc[0]

        for cand_id, source, score in cand_list:
            if source == 'S2' and cand_id in s2_lookup.index:
                cand_row = s2_lookup.loc[cand_id]
            elif source == 'S3' and cand_id in s3_lookup.index:
                cand_row = s3_lookup.loc[cand_id]
            else:
                continue

            feats = extract_features_pair(
                str(s1_row['business_name']),
                str(s1_row['business_address']),
                str(s1_row['country']),
                str(cand_row['business_name']),
                str(cand_row['business_address']),
                str(cand_row['country']),
            )
            # Add country info for frequency feature
            feats['s1_country'] = normalize_country(str(s1_row['country']))
            feats['candidate_country'] = normalize_country(str(cand_row['country']))
            # (Will compute frequency later)


            feats['candidate_id'] = cand_id
            feats['s1_id'] = s1_id
            feats['source'] = source
            feats['blocking_score'] = score
            feature_rows.append(feats)

            # Label
            true_matches = gt_dict.get(s1_id, [])
            labels.append(1 if cand_id in true_matches else 0)

    features_df = pd.DataFrame(feature_rows)
    labels = np.array(labels, dtype=np.int32)
    print(f"Total candidate pairs: {len(features_df)}")
    print(f"Positive pairs: {labels.sum()}, Negative pairs: {len(labels) - labels.sum()}")
    print(f"Positive rate: {labels.mean():.4f}")

    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS + ['blocking_score', 'source']:
        if col not in features_df.columns:
            features_df[col] = 0.0

    return features_df, labels


def optimize_threshold(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    val_s1_ids: pd.Series,
    gt_dict: Dict[str, List[str]],
    n_thresholds: int = 100,
) -> Tuple[float, dict]:
    """
    Find the probability threshold that maximizes Macro F0.5 on validation data.
    """
    # Get predicted probabilities
    full_feature_cols = FEATURE_COLUMNS + ['blocking_score', 'source']
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_val[full_feature_cols])[:, 1]
    else:
        probs = model.predict(X_val[full_feature_cols])

    # Build predictions dict for threshold optimization
    val_pairs = pd.DataFrame({
        's1_id': val_s1_ids.values,
        'candidate_id': X_val['candidate_id'].values,
        'source': X_val['source'].values,
        'probability': probs,
    })

    best_threshold = 0.5
    best_f05 = 0.0
    best_stats = {}

    # Try different thresholds
    thresholds = np.linspace(0.05, 0.95, n_thresholds)

    for threshold in thresholds:
        # Group predictions by S1 entity
        predictions = {}
        above_threshold = val_pairs[val_pairs['probability'] >= threshold]
        for s1_id, group in above_threshold.groupby('s1_id'):
            predictions[s1_id] = list(group['candidate_id'])

        # Ensure all S1 entities in validation set have an entry
        for s1_id in val_s1_ids:
            if s1_id not in predictions:
                predictions[s1_id] = []

        # Compute Macro F0.5
        stats = compute_macro_f05(gt_dict, predictions)
        if stats['macro_f05'] > best_f05:
            best_f05 = stats['macro_f05']
            best_threshold = threshold
            best_stats = stats

    return best_threshold, best_stats


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    val_s1_ids: pd.Series,
    val_gt_dict: Dict[str, List[str]],
) -> Tuple:
    """
    Train LightGBM model and optimize threshold.
    """
    # Calculate scale_pos_weight for class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1) * 0.5  # Reduce weight to avoid over-predicting

    print(f"Training with {len(X_train)} samples, pos: {n_pos}, neg: {n_neg}, scale_pos_weight: {scale_pos_weight:.2f}")

    # LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # Train
    feature_cols = FEATURE_COLUMNS + ['blocking_score', 'source']
    model.fit(
        X_train[feature_cols],
        y_train,
        eval_set=[(X_val[feature_cols], y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Optimize threshold
    best_threshold, best_stats = optimize_threshold(
        model, X_val, y_val, val_s1_ids, val_gt_dict
    )

    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best validation F0.5: {best_stats['macro_f05']:.4f}")
    print(f"Validation precision: {best_stats['macro_precision']:.4f}")
    print(f"Validation recall: {best_stats['macro_recall']:.4f}")

    return model, best_threshold, best_stats


if __name__ == '__main__':
    model, threshold, vectorizer, stats = run_training_pipeline(
        n_train_samples=5000,
        n_val_samples=3000,
        cosine_threshold=0.05,
        top_k=50,
    )
    print("\nTraining complete!")
    print(f"Best F0.5: {stats['macro_f05']:.4f}")
