"""
Prediction module for Business Entity Resolution.

Loads trained model and artifacts, generates candidate pairs for the test
set, scores them, and writes output files (candidate_pairs.tsv and
matching_results.tsv).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.sparse import vstack
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_source1, load_source2, load_source3, DATASET_DIR
from preprocess import normalize_name_compact, normalize_name_compact_series, normalize_country, normalize_country_series, normalize_address
from blocking import (
    compute_tfidf_matrix, build_country_groups,
    BATCH_SIZE, COSINE_THRESHOLD, TOP_K_CANDIDATES, CANDIDATE_BATCH_SIZE
)
from feature_extraction import extract_features_pair, FEATURE_COLUMNS, set_country_pair_freq
from train_matching import compute_macro_f05

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
SCRATCH_DIR = os.path.join(BASE_DIR, 'scratch')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def load_artifacts():
    """Load trained model, vectorizer, threshold, and feature list."""
    model_path = os.path.join(SCRATCH_DIR, 'models', 'lgbm_model.pkl')
    vectorizer_path = os.path.join(SCRATCH_DIR, 'models', 'tfidf_vectorizer.pkl')
    threshold_path = os.path.join(SCRATCH_DIR, 'models', 'threshold.txt')
    features_path = os.path.join(SCRATCH_DIR, 'models', 'feature_columns.txt')
    freq_path = os.path.join(SCRATCH_DIR, 'models', 'country_pair_freq.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())

    feature_cols = []
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            feature_cols = [line.strip() for line in f if line.strip()]

    # Load country-pair frequency table
    freq_table = {}
    if os.path.exists(freq_path):
        with open(freq_path, 'rb') as f:
            freq_table = pickle.load(f)
        set_country_pair_freq(freq_table)
        print(f"Loaded country-pair freq: {len(freq_table)} entries")

    print(f"Loaded model from {model_path}")
    print(f"Loaded vectorizer from {vectorizer_path}")
    print(f"Threshold: {threshold}")
    print(f"Feature columns: {len(feature_cols)}")

    return model, vectorizer, threshold, feature_cols


def predict_for_country(
    s1_group: pd.DataFrame,
    s2_group: pd.DataFrame,
    s3_group: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    model,
    threshold: float,
    feature_cols: List[str],
    cosine_threshold: float = 0.15,
    top_k: int = 30,
    fast_mode: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate candidates and predictions for a specific country group.

    Returns:
        candidates_dict: S1 ID -> list of candidate IDs
        predictions_dict: S1 ID -> list of predicted match IDs
    """
    candidates_dict = {}
    predictions_dict = {}

    if len(s1_group) == 0:
        return candidates_dict, predictions_dict

    # Preprocess names
    s1_names = normalize_name_compact_series(s1_group['business_name']).tolist()
    s2_names = normalize_name_compact_series(s2_group['business_name']).tolist()
    s3_names = normalize_name_compact_series(s3_group['business_name']).tolist()

    # TF-IDF matrices (L2 normalized for cosine via dot product)
    s1_matrix = compute_tfidf_matrix(vectorizer, s1_names)
    s2_matrix = compute_tfidf_matrix(vectorizer, s2_names)
    s3_matrix = compute_tfidf_matrix(vectorizer, s3_names)

    # Combine S2 and S3
    if len(s2_group) > 0 and len(s3_group) > 0:
        combined_matrix = vstack([s2_matrix, s3_matrix])
        combined_ids = list(s2_group['entity_id']) + list(s3_group['entity_id'])
        combined_sources = ['S2'] * len(s2_group) + ['S3'] * len(s3_group)
        combined_names = s2_names + s3_names
        combined_addrs = list(s2_group['business_address']) + list(s3_group['business_address'])
        combined_countries = list(s2_group['country']) + list(s3_group['country'])
    elif len(s2_group) > 0:
        combined_matrix = s2_matrix
        combined_ids = list(s2_group['entity_id'])
        combined_sources = ['S2'] * len(s2_group)
        combined_names = s2_names
        combined_addrs = list(s2_group['business_address'])
        combined_countries = list(s2_group['country'])
    elif len(s3_group) > 0:
        combined_matrix = s3_matrix
        combined_ids = list(s3_group['entity_id'])
        combined_sources = ['S3'] * len(s3_group)
        combined_names = s3_names
        combined_addrs = list(s3_group['business_address'])
        combined_countries = list(s3_group['country'])
    else:
        return candidates_dict, predictions_dict

    n_candidates = len(combined_ids)
    source_map = {'S2': 0, 'S3': 1}

    # Process S1 in batches, then candidates in sub-batches to limit memory
    for start in range(0, len(s1_group), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(s1_group))
        batch_s1_matrix = s1_matrix[start:end]
        batch_s1_ids = list(s1_group['entity_id'].iloc[start:end])
        batch_s1_names = s1_names[start:end]
        batch_s1_addrs = list(s1_group['business_address'].iloc[start:end])
        batch_s1_countries = list(s1_group['country'].iloc[start:end])
        n_batch = len(batch_s1_ids)

        # Initialize top-K accumulator per S1 entity
        top_scores = [[] for _ in range(n_batch)]  # list of (score, global_idx)

        # Process candidates in sub-batches
        for cstart in range(0, n_candidates, CANDIDATE_BATCH_SIZE):
            cend = min(cstart + CANDIDATE_BATCH_SIZE, n_candidates)
            cand_batch = combined_matrix[cstart:cend]
            sims = batch_s1_matrix.dot(cand_batch.T).toarray()  # (n_batch, cend-cstart)

            for i in range(n_batch):
                sim_scores = sims[i]
                c_indices = np.where(sim_scores >= cosine_threshold)[0]
                if len(c_indices) == 0:
                    continue
                global_indices = cstart + c_indices
                scores = sim_scores[c_indices]
                top_scores[i].extend(zip(scores, global_indices))
                if len(top_scores[i]) > top_k:
                    top_scores[i] = sorted(top_scores[i], key=lambda x: -x[0])[:top_k]

        for i, s1_id in enumerate(batch_s1_ids):
            # Sort by score descending and take top_k
            sorted_cands = sorted(top_scores[i], key=lambda x: -x[0])[:top_k]

            # Collect candidate info
            cand_info = []
            for score, idx in sorted_cands:
                cand_info.append({
                    'id': combined_ids[idx],
                    'source': combined_sources[idx],
                    'score': float(score),
                    'name': combined_names[idx],
                    'addr': combined_addrs[idx],
                    'country': combined_countries[idx],
                })

            # Token-based fallback: if no TF-IDF candidates, check token overlap
            if len(cand_info) == 0:
                s1_tokens = set(extract_name_tokens(batch_s1_names[i]))
                if s1_tokens:
                    for j, cname in enumerate(combined_names):
                        ctokens = set(extract_name_tokens(cname))
                        if len(s1_tokens & ctokens) >= 2:
                            cand_info.append({
                                'id': combined_ids[j],
                                'source': combined_sources[j],
                                'score': 0.0,
                                'name': cname,
                                'addr': combined_addrs[j],
                                'country': combined_countries[j],
                            })
                            if len(cand_info) >= top_k:
                                break

            if len(cand_info) == 0:
                candidates_dict[s1_id] = []
                predictions_dict[s1_id] = []
                continue

            cand_ids = [c['id'] for c in cand_info]
            candidates_dict[s1_id] = cand_ids

            # In fast mode, use TF-IDF cosine score directly
            if fast_mode:
                matched_ids = [cand_ids[j] for j in range(len(cand_ids)) if cand_info[j]['score'] >= threshold]
                predictions_dict[s1_id] = matched_ids
                continue

            # Extract features and score with model
            s1_name = batch_s1_names[i]
            s1_addr = batch_s1_addrs[i]
            s1_country = batch_s1_countries[i]

            feature_rows = []
            for c in cand_info:
                feats = extract_features_pair(
                    s1_name, s1_addr, s1_country,
                    c['name'], c['addr'], c['country']
                )
                feats['blocking_score'] = c['score']
                feats['source'] = source_map.get(c['source'], 0)
                feature_rows.append(feats)

            X = pd.DataFrame(feature_rows)

            # Add source column for model
            X['source'] = X['source'].astype(float)
            X['blocking_score'] = X['blocking_score'].astype(float)

            # Predict
            probs = model.predict_proba(X[feature_cols])[:, 1]

            # Apply threshold and generate predictions
            matched_ids = [cand_ids[j] for j in range(len(cand_ids)) if probs[j] >= threshold]
            predictions_dict[s1_id] = matched_ids

        if (start // BATCH_SIZE + 1) % 5 == 0:
            print(f"  Processed {end}/{len(s1_group)} S1 entities for country {s1_group.iloc[0]['country_norm']}")

    return candidates_dict, predictions_dict


def extract_name_tokens(name: str) -> List[str]:
    """Extract significant tokens from a name."""
    if not name:
        return []
    tokens = name.split()
    return [t for t in tokens if len(t) > 2]


def run_prediction(
    cosine_threshold: float = None,
    top_k: int = None,
    score_threshold: float = None,
    test_s1_nrows: int = None,
    test_s2_nrows: int = None,
    test_s3_nrows: int = None,
    fast_mode: bool = False,
):
    """
    Run the full prediction pipeline on the test set.

    Generates candidate_pairs.tsv and matching_results.tsv.
    """
    print("=" * 60)
    print("Loading trained artifacts...")
    print("=" * 60)

    model, vectorizer, default_threshold, feature_cols = load_artifacts()

    if score_threshold is not None:
        threshold = score_threshold
    else:
        threshold = default_threshold

    cosine_threshold = cosine_threshold or COSINE_THRESHOLD
    top_k = top_k or TOP_K_CANDIDATES

    print(f"\nUsing score threshold: {threshold}")

    print("\n" + "=" * 60)
    print("Loading test data...")
    print("=" * 60)

    if test_s1_nrows is not None:
        s1_test = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source1.tsv'),
                              sep='\t', encoding='utf-8', dtype=str,
                              keep_default_na=False, nrows=test_s1_nrows)
        s2_test = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source2.tsv'),
                              sep='\t', encoding='utf-8', dtype=str,
                              keep_default_na=False, nrows=test_s2_nrows if test_s2_nrows else test_s1_nrows)
        s3_test = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source3.tsv'),
                              sep='\t', encoding='utf-8', dtype=str,
                              keep_default_na=False, nrows=test_s3_nrows if test_s3_nrows else test_s1_nrows)
        for df in [s1_test, s2_test, s3_test]:
            for col in ['business_name', 'business_address', 'country']:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
    else:
        s1_test = load_source1('test')
        s2_test = load_source2('test')
        s3_test = load_source3('test')

    print(f"Test S1: {len(s1_test)}, Test S2: {len(s2_test)}, Test S3: {len(s3_test)}")

    # Normalize country
    s1_test['country_norm'] = normalize_country_series(s1_test['country'])
    s2_test['country_norm'] = normalize_country_series(s2_test['country'])
    s3_test['country_norm'] = normalize_country_series(s3_test['country'])

    all_candidates = {}
    all_predictions = {}

    countries = sorted(set(s1_test['country_norm'].unique()) |
                       set(s2_test['country_norm'].unique()) |
                       set(s3_test['country_norm'].unique()))

    print(f"\nCountries: {countries}")

    for country in countries:
        print(f"\n{'='*60}")
        print(f"Processing country: {country}")
        print(f"{'='*60}")

        s1_g = s1_test[s1_test['country_norm'] == country].reset_index(drop=True)
        s2_g = s2_test[s2_test['country_norm'] == country].reset_index(drop=True)
        s3_g = s3_test[s3_test['country_norm'] == country].reset_index(drop=True)

        print(f"S1: {len(s1_g)}, S2: {len(s2_g)}, S3: {len(s3_g)}")

        if len(s1_g) == 0:
            continue

        cand_dict, pred_dict = predict_for_country(
            s1_g, s2_g, s3_g,
            vectorizer, model, threshold, feature_cols,
            cosine_threshold=cosine_threshold,
            top_k=top_k,
            fast_mode=fast_mode,
        )

        all_candidates.update(cand_dict)
        all_predictions.update(pred_dict)

        print(f"Country {country}: {len(cand_dict)} S1 entities, "
              f"{sum(len(v) for v in cand_dict.values())} total candidates, "
              f"{sum(len(v) for v in pred_dict.values())} total predictions")

    # Ensure every S1 entity has an entry
    for s1_id in s1_test['entity_id']:
        if s1_id not in all_candidates:
            all_candidates[s1_id] = []
        if s1_id not in all_predictions:
            all_predictions[s1_id] = []

    # Write output files
    print("\n" + "=" * 60)
    print("Writing output files...")
    print("=" * 60)

    # candidate_pairs.tsv
    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in s1_test['entity_id']:
            cands = all_candidates.get(s1_id, [])
            f.write(f'{s1_id}\t{",".join(cands) if cands else ""}\n')

    # matching_results.tsv
    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in s1_test['entity_id']:
            matches = all_predictions.get(s1_id, [])
            f.write(f'{s1_id}\t{",".join(matches) if matches else ""}\n')

    print(f"Candidate pairs written to: {cand_path}")
    print(f"Matching results written to: {match_path}")

    # Print statistics
    total_cands = sum(len(v) for v in all_candidates.values())
    total_matches = sum(len(v) for v in all_predictions.values())
    print(f"\nTotal S1 entities: {len(s1_test)}")
    print(f"Total candidate pairs: {total_cands}")
    print(f"Total predicted matches: {total_matches}")
    print(f"Avg candidates per S1: {total_cands / len(s1_test):.1f}")
    print(f"Avg matches per S1: {total_matches / len(s1_test):.4f}")
    print(f"S1 entities with matches: {sum(1 for v in all_predictions.values() if len(v) > 0)}")

    return all_candidates, all_predictions


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cosine-threshold', type=float, default=0.15)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--score-threshold', type=float, default=None)
    parser.add_argument('--s1-nrows', type=int, default=None)
    parser.add_argument('--s2-nrows', type=int, default=None)
    parser.add_argument('--s3-nrows', type=int, default=None)
    parser.add_argument('--fast-mode', action='store_true', help='Skip feature extraction, use TF-IDF cosine only')
    args = parser.parse_args()

    run_prediction(
        cosine_threshold=args.cosine_threshold,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        test_s1_nrows=args.s1_nrows,
        test_s2_nrows=args.s2_nrows,
        test_s3_nrows=args.s3_nrows,
        fast_mode=args.fast_mode,
    )
