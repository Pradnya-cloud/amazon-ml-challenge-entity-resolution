"""
Fast standalone prediction script for hackathon.
Uses TF-IDF cosine similarity for candidate generation and the trained model for filtering.
Optimized for speed with small row limits.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
SCRATCH_DIR = os.path.join(BASE_DIR, 'scratch')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

sys.path.insert(0, _THIS_DIR)

from preprocess import normalize_name_compact_series, normalize_country_series
from data_loader import DATASET_DIR


def load_test_subset(s1_nrows=30000, s2_nrows=50000, s3_nrows=50000):
    s1 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source1.tsv'),
                     sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False, nrows=s1_nrows)
    s2 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source2.tsv'),
                     sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False, nrows=s2_nrows)
    s3 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source3.tsv'),
                     sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False, nrows=s3_nrows)

    for df in [s1, s2, s3]:
        for col in ['business_name', 'business_address', 'country']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
    return s1, s2, s3


def main():
    print("Loading artifacts...")

    with open(os.path.join(SCRATCH_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(SCRATCH_DIR, 'models', 'lgbm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(SCRATCH_DIR, 'models', 'threshold.txt'), 'r') as f:
        threshold = float(f.read().strip())

    with open(os.path.join(SCRATCH_DIR, 'models', 'feature_columns.txt'), 'r') as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    from feature_extraction import extract_features_pair, FEATURE_COLUMNS
    set_cols = FEATURE_COLUMNS + ['blocking_score', 'source']
    for col in set_cols:
        if col not in feature_cols:
            feature_cols = FEATURE_COLUMNS + ['blocking_score', 'source']
            break

    print(f"Threshold: {threshold}, Feature cols: {len(feature_cols)}")

    # Load test data (small subset)
    s1, s2, s3 = load_test_subset(s1_nrows=20000, s2_nrows=30000, s3_nrows=30000)
    print(f"Test S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")

    # Normalize countries
    s1['country_norm'] = normalize_country_series(s1['country'])
    s2['country_norm'] = normalize_country_series(s2['country'])
    s3['country_norm'] = normalize_country_series(s3['country'])

    s1_names = normalize_name_compact_series(s1['business_name']).tolist()
    s2_names = normalize_name_compact_series(s2['business_name']).tolist()
    s3_names = normalize_name_compact_series(s3['business_name']).tolist()

    s1_matrix = vectorizer.transform(s1_names)
    s2_matrix = vectorizer.transform(s2_names)
    s3_matrix = vectorizer.transform(s3_names)

    BATCH = 500
    CAND_BATCH = 2000
    COS_SIM_THRESHOLD = 0.3
    TOP_K = 5

    all_candidates = {}
    all_predictions = {}

    countries = sorted(set(s1['country_norm'].unique()) |
                      set(s2['country_norm'].unique()) |
                      set(s3['country_norm'].unique()))
    print(f"Countries: {countries}")

    for country in countries:
        s1_mask = s1['country_norm'].values == country
        s2_mask = s2['country_norm'].values == country
        s3_mask = s3['country_norm'].values == country

        s1_idx = np.where(s1_mask)[0]
        if len(s1_idx) == 0:
            continue

        cand_ids = list(s2['entity_id'][s2_mask]) + list(s3['entity_id'][s3_mask])
        cand_sources = ['S2'] * s2_mask.sum() + ['S3'] * s3_mask.sum()
        cand_names = [n for n, m in zip(s2_names, s2_mask) if m] + [n for n, m in zip(s3_names, s3_mask) if m]
        cand_addrs = list(s2['business_address'][s2_mask]) + list(s3['business_address'][s3_mask])
        cand_countries = list(s2['country'][s2_mask]) + list(s3['country'][s3_mask])

        if len(cand_ids) == 0:
            for idx in s1_idx:
                s1_id = s1['entity_id'].iloc[idx]
                all_candidates[s1_id] = []
                all_predictions[s1_id] = []
            continue

        cand_matrix = vstack([s2_matrix[s2_mask], s3_matrix[s3_mask]])
        s1_sub = s1_matrix[s1_idx]
        n_s1c = len(s1_idx)
        n_candc = len(cand_ids)
        print(f"  {country}: S1={n_s1c}, Candidates={n_candc}")

        for start in range(0, n_s1c, BATCH):
            end = min(start + BATCH, n_s1c)
            batch_mat = s1_sub[start:end]
            batch_ids = list(s1['entity_id'].iloc[s1_idx[start:end]])

            top_per_s1 = [[] for _ in range(end - start)]

            for cstart in range(0, n_candc, CAND_BATCH):
                cend = min(cstart + CAND_BATCH, n_candc)
                sims = batch_mat.dot(cand_matrix[cstart:cend].T).toarray()

                for i in range(end - start):
                    scores = sims[i]
                    idx_above = np.where(scores >= COS_SIM_THRESHOLD)[0]
                    for local_idx in idx_above:
                        top_per_s1[i].append((float(scores[local_idx]), cstart + local_idx))
                    # Trim early to limit memory and sorting time
                    if len(top_per_s1[i]) > TOP_K * 5:
                        top_per_s1[i] = sorted(top_per_s1[i], key=lambda x: -x[0])[:TOP_K * 2]

            for i, s1_id in enumerate(batch_ids):
                sorted_cands = sorted(top_per_s1[i], key=lambda x: -x[0])[:TOP_K]
                cand_ids_batch = [cand_ids[idx] for _, idx in sorted_cands]
                all_candidates[s1_id] = cand_ids_batch

                if sorted_cands:
                    feature_rows = []
                    s1_row = s1.iloc[s1_idx[start + i]]
                    s1_name = s1_names[s1_idx[start + i]]
                    s1_addr = str(s1_row['business_address'])
                    s1_country = str(s1_row['country'])

                    for score, idx in sorted_cands:
                        feats = extract_features_pair(
                            s1_name, s1_addr, s1_country,
                            cand_names[idx], cand_addrs[idx], cand_countries[idx]
                        )
                        feats['blocking_score'] = float(score)
                        feats['source'] = 0.0 if cand_sources[idx] == 'S2' else 1.0
                        feats['candidate_id'] = cand_ids[idx]
                        feature_rows.append(feats)

                    X = pd.DataFrame(feature_rows)
                    for col in set_cols:
                        if col in X.columns:
                            X[col] = X[col].astype(float)
                    X = X[set_cols]
                    probs = model.predict_proba(X)[:, 1]
                    matched = [cand_ids[idx] for j, (_, idx) in enumerate(sorted_cands) if probs[j] >= threshold]
                    all_predictions[s1_id] = matched
                else:
                    all_predictions[s1_id] = []

        print(f"  {country}: predictions={sum(len(v) for v in all_predictions.values() if v)}")

    # Ensure all S1 entities have entries
    for s1_id in s1['entity_id']:
        if s1_id not in all_candidates:
            all_candidates[s1_id] = []
        if s1_id not in all_predictions:
            all_predictions[s1_id] = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in s1['entity_id']:
            cands = all_candidates.get(s1_id, [])
            f.write(f'{s1_id}\t{",".join(cands) if cands else ""}\n')

    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in s1['entity_id']:
            matches = all_predictions.get(s1_id, [])
            f.write(f'{s1_id}\t{",".join(matches) if matches else ""}\n')

    total_cands = sum(len(v) for v in all_candidates.values())
    total_matches = sum(len(v) for v in all_predictions.values())
    print(f"\nDone! S1: {len(s1)}, Candidates: {total_cands}, Matches: {total_matches}")
    print(f"S1 with matches: {sum(1 for v in all_predictions.values() if len(v) > 0)}")


if __name__ == '__main__':
    main()
