"""
Full-test prediction using TF-IDF cosine similarity only.
No per-pair feature extraction for speed.
Processes full 1.7M test S1 entities efficiently.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import vstack

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
sys.path.insert(0, _THIS_DIR)

from data_loader import DATASET_DIR
from preprocess import normalize_name_compact_series, normalize_country_series
from feature_extraction import extract_features_pair, FEATURE_COLUMNS
import lightgbm as lgb

OUTPUT_DIR = os.path.join(_BASE_DIR, 'output')
SCRATCH_DIR = os.path.join(_BASE_DIR, 'scratch')


def load_artifacts():
    with open(os.path.join(SCRATCH_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    model_path = os.path.join(SCRATCH_DIR, 'models', 'lgbm_model.pkl')
    threshold_path = os.path.join(SCRATCH_DIR, 'models', 'threshold.txt')
    model = None
    threshold = 0.5
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
    feature_cols = FEATURE_COLUMNS + ['blocking_score', 'source']
    return model, vectorizer, threshold, feature_cols


def load_full_test():
    print("Loading all test data...")
    s1 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source1.tsv'),
                     sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False)
    s2 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source2.tsv'),
                     sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False)
    s3 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source3.tsv'),
                     sep='\t', encoding='utf-8', dtype=str,
                     keep_default_na=False)
    for df in [s1, s2, s3]:
        for col in ['business_name', 'business_address', 'country']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
    print(f"Loaded S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")
    return s1, s2, s3


def run_full_prediction(batch_size=5000, cos_threshold=0.6, top_k=3, use_model=True):
    model, vectorizer, model_threshold, feature_cols = load_artifacts()
    s1, s2, s3 = load_full_test()

    s1['country_norm'] = normalize_country_series(s1['country'])
    s2['country_norm'] = normalize_country_series(s2['country'])
    s3['country_norm'] = normalize_country_series(s3['country'])

    print("Computing TF-IDF matrices...")
    s1_names = normalize_name_compact_series(s1['business_name']).tolist()
    s2_names = normalize_name_compact_series(s2['business_name']).tolist()
    s3_names = normalize_name_compact_series(s3['business_name']).tolist()

    s1_matrix = vectorizer.transform(s1_names)
    s2_matrix = vectorizer.transform(s2_names)
    s3_matrix = vectorizer.transform(s3_names)
    print(f"S1: {s1_matrix.shape}, S2: {s2_matrix.shape}, S3: {s3_matrix.shape}")

    countries = sorted(set(s1['country_norm'].unique()) |
                      set(s2['country_norm'].unique()) |
                      set(s3['country_norm'].unique()))
    print(f"Countries: {countries}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')

    cand_f = open(cand_path, 'w', encoding='utf-8')
    match_f = open(match_path, 'w', encoding='utf-8')
    cand_f.write('source1_entity_id\tcandidate_entity_ids\n')
    match_f.write('source1_entity_id\tmatched_entity_ids\n')

    total_s1 = 0
    total_matches = 0
    CAND_BATCH = 10000

    for country in countries:
        print(f"\nProcessing country: {country}")
        s1_mask = s1['country_norm'].values == country
        s2_mask = s2['country_norm'].values == country
        s3_mask = s3['country_norm'].values == country

        s1_indices = np.where(s1_mask)[0]
        n_s1c = len(s1_indices)
        if n_s1c == 0:
            continue

        s2_indices = np.where(s2_mask)[0]
        s3_indices = np.where(s3_mask)[0]

        s1_sub = s1_matrix[s1_indices]

        if len(s2_indices) > 0 and len(s3_indices) > 0:
            comb_matrix = vstack([s2_matrix[s2_indices], s3_matrix[s3_indices]])
            comb_ids = list(s2['entity_id'].iloc[s2_indices]) + list(s3['entity_id'].iloc[s3_indices])
            comb_sources = ['S2'] * len(s2_indices) + ['S3'] * len(s3_indices)
            comb_names = [s2_names[i] for i in s2_indices] + [s3_names[i] for i in s3_indices]
            comb_addrs = list(s2['business_address'].iloc[s2_indices]) + list(s3['business_address'].iloc[s3_indices])
            comb_countries = list(s2['country'].iloc[s2_indices]) + list(s3['country'].iloc[s3_indices])
        elif len(s2_indices) > 0:
            comb_matrix = s2_matrix[s2_indices]
            comb_ids = list(s2['entity_id'].iloc[s2_indices])
            comb_sources = ['S2'] * len(s2_indices)
            comb_names = [s2_names[i] for i in s2_indices]
            comb_addrs = list(s2['business_address'].iloc[s2_indices])
            comb_countries = list(s2['country'].iloc[s2_indices])
        elif len(s3_indices) > 0:
            comb_matrix = s3_matrix[s3_indices]
            comb_ids = list(s3['entity_id'].iloc[s3_indices])
            comb_sources = ['S3'] * len(s3_indices)
            comb_names = [s3_names[i] for i in s3_indices]
            comb_addrs = list(s3['business_address'].iloc[s3_indices])
            comb_countries = list(s3['country'].iloc[s3_indices])
        else:
            for idx in s1_indices:
                s1_id = s1['entity_id'].iloc[idx]
                cand_f.write(f'{s1_id}\t\n')
                match_f.write(f'{s1_id}\t\n')
                total_s1 += 1
            continue

        n_cand = len(comb_ids)
        print(f"  S1: {n_s1c}, Candidates: {n_cand}")

        for start in range(0, n_s1c, batch_size):
            end = min(start + batch_size, n_s1c)
            batch_mat = s1_sub[start:end]
            batch_s1_ids = list(s1['entity_id'].iloc[s1_indices[start:end]])
            batch_s1_names = [s1_names[s1_indices[start + i]] for i in range(end - start)]
            batch_s1_addrs = list(s1['business_address'].iloc[s1_indices[start:end]])
            batch_s1_countries = list(s1['country'].iloc[s1_indices[start:end]])

            # Collect top-K candidates using sparse matrix multiplication
            top_per_s1 = [[] for _ in range(end - start)]

            for cstart in range(0, n_cand, CAND_BATCH):
                cend = min(cstart + CAND_BATCH, n_cand)
                sims = batch_mat.dot(comb_matrix[cstart:cend].T).toarray()

                for i in range(end - start):
                    sim_scores = sims[i]
                    # Use argpartition for fast top-K
                    if sim_scores.size > top_k:
                        top_idx = np.argpartition(sim_scores, -top_k)[-top_k:]
                    else:
                        top_idx = np.arange(sim_scores.size)
                    top_scores = sim_scores[top_idx]
                    for j, local_idx in enumerate(top_idx):
                        s = float(top_scores[j])
                        if s >= cos_threshold:
                            top_per_s1[i].append((s, cstart + local_idx))
                    # Trim
                    if len(top_per_s1[i]) > top_k:
                        top_per_s1[i] = sorted(top_per_s1[i], key=lambda x: -x[0])[:top_k]

            for i, s1_id in enumerate(batch_s1_ids):
                sorted_cands = sorted(top_per_s1[i], key=lambda x: -x[0])[:top_k]
                cand_ids = [comb_ids[idx] for _, idx in sorted_cands]
                cand_f.write(f'{s1_id}\t{",".join(cand_ids) if cand_ids else ""}\n')

                if not sorted_cands:
                    match_f.write(f'{s1_id}\t\n')
                    total_s1 += 1
                    continue

                if use_model and model is not None:
                    feature_rows = []
                    s1_name = batch_s1_names[i]
                    s1_addr = batch_s1_addrs[i]
                    s1_country = batch_s1_countries[i]

                    for score, idx in sorted_cands:
                        feats = extract_features_pair(
                            s1_name, s1_addr, s1_country,
                            comb_names[idx], comb_addrs[idx], comb_countries[idx]
                        )
                        feats['blocking_score'] = float(score)
                        feats['source'] = 0.0 if comb_sources[idx] == 'S2' else 1.0
                        feature_rows.append(feats)

                    X = pd.DataFrame(feature_rows)
                    for col in feature_cols:
                        if col in X.columns:
                            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
                    X = X[feature_cols]
                    probs = model.predict_proba(X)[:, 1]
                    matched = [comb_ids[idx] for j, (_, idx) in enumerate(sorted_cands) if probs[j] >= model_threshold]
                else:
                    # Use cosine score directly as match score
                    matched = [comb_ids[idx] for score, idx in sorted_cands if score >= cos_threshold]

                match_f.write(f'{s1_id}\t{",".join(matched) if matched else ""}\n')
                total_s1 += 1
                total_matches += len(matched)

            if (start // batch_size + 1) % 20 == 0:
                print(f"  Processed {end}/{n_s1c} for {country}, matches: {total_matches}")

        print(f"  {country}: done ({n_s1c} entities)")

    cand_f.close()
    match_f.close()
    print(f"\nDone! Total S1: {total_s1}, Total matches: {total_matches}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--cos-threshold', type=float, default=0.6)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--no-model', action='store_true', help='Skip model inference, use TF-IDF cosine only')
    args = parser.parse_args()

    run_full_prediction(
        batch_size=args.batch_size,
        cos_threshold=args.cos_threshold,
        top_k=args.top_k,
        use_model=not args.no_model,
    )
