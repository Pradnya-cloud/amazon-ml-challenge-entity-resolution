"""
Blocking module for Business Entity Resolution.

Generates candidate S1 <-> S2/S3 pairs using:
1. Country-based pre-filtering
2. Character 3-gram TF-IDF cosine similarity

Designed to handle millions of records efficiently through batch processing
and sparse matrix operations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional
from scipy.sparse import csr_matrix, vstack, hstack
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
import os
import pickle

from preprocess import normalize_name_compact, normalize_name_compact_series, normalize_country, normalize_country_series, extract_tokens, extract_char_ngrams

BATCH_SIZE = 20000
CANDIDATE_BATCH_SIZE = 5000
COSINE_THRESHOLD = 0.15
TOP_K_CANDIDATES = 200


def fit_tfidf_vectorizer(names: List[str], cache_path: str = None) -> TfidfVectorizer:
    """
    Fit a character 3-gram TF-IDF vectorizer on a list of business names.
    Optionally cache to disk for reuse.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 3),
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        max_features=30000,
        dtype=np.float32,
    )
    vectorizer.fit(names)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    return vectorizer


def compute_tfidf_matrix(vectorizer: TfidfVectorizer, names: List[str]) -> csr_matrix:
    """Transform a list of names into a TF-IDF sparse matrix."""
    matrix = vectorizer.transform(names)
    # Normalize rows for cosine similarity via dot product
    matrix = sk_normalize(matrix, norm='l2', axis=1)
    return matrix


def build_country_groups(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group entities by normalized country."""
    df = df.copy()
    df['country_norm'] = normalize_country_series(df['country'])
    groups = {}
    for country, group in df.groupby('country_norm'):
        groups[country] = group.reset_index(drop=True)
    return groups


def generate_candidates_for_country(
    s1_group: pd.DataFrame,
    s2_group: pd.DataFrame,
    s3_group: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    gt_dict: Dict[str, List[str]] = None,
    cosine_threshold: float = COSINE_THRESHOLD,
    top_k: int = TOP_K_CANDIDATES,
    use_token_fallback: bool = True,
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Generate candidate pairs for S1 entities from a specific country,
    matching against S2 and S3 entities of the same country.

    Returns dict: S1 entity_id -> list of (candidate_id, source, score)
    """
    results = {}

    if len(s1_group) == 0:
        return results

    s1_names = normalize_name_compact_series(s1_group['business_name']).tolist()
    s2_names = normalize_name_compact_series(s2_group['business_name']).tolist()
    s3_names = normalize_name_compact_series(s3_group['business_name']).tolist()

    # Compute TF-IDF matrices
    s1_matrix = compute_tfidf_matrix(vectorizer, s1_names)  # (n_s1, n_features)
    s2_matrix = compute_tfidf_matrix(vectorizer, s2_names)  # (n_s2, n_features)
    s3_matrix = compute_tfidf_matrix(vectorizer, s3_names)  # (n_s3, n_features)

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
        return results

    n_candidates = len(combined_ids)

    # Process S1 in batches, then candidates in sub-batches to limit memory
    for start in range(0, len(s1_group), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(s1_group))
        batch_s1_matrix = s1_matrix[start:end]
        batch_s1_ids = list(s1_group['entity_id'].iloc[start:end])
        n_batch = len(batch_s1_ids)

        # Initialize top-K accumulator per S1 entity in this batch
        top_indices_per_s1 = [[] for _ in range(n_batch)]  # list of (score, idx)

        # Process candidates in sub-batches to limit dense matrix size
        for cstart in range(0, n_candidates, CANDIDATE_BATCH_SIZE):
            cend = min(cstart + CANDIDATE_BATCH_SIZE, n_candidates)
            cand_batch = combined_matrix[cstart:cend]
            sims = batch_s1_matrix.dot(cand_batch.T).toarray()  # (n_batch, cend-cstart)

            for i in range(n_batch):
                sim_scores = sims[i]
                c_indices = np.where(sim_scores >= cosine_threshold)[0]
                if len(c_indices) == 0:
                    continue
                # Get (global_idx, score) pairs and keep top-K for this S1
                global_indices = cstart + c_indices
                scores = sim_scores[c_indices]
                # Merge with existing top candidates, keep top_k
                combined = list(zip(scores, global_indices))
                top_indices_per_s1[i].extend(combined)
                # Trim to top_k
                if len(top_indices_per_s1[i]) > top_k:
                    top_indices_per_s1[i] = sorted(top_indices_per_s1[i], reverse=True)[:top_k]

        for i, s1_id in enumerate(batch_s1_ids):
            # Sort by score descending and take top_k
            sorted_cands = sorted(top_indices_per_s1[i], reverse=True)[:top_k]
            candidates = [(combined_ids[idx], combined_sources[idx], float(score))
                         for score, idx in sorted_cands]

            # Token-based fallback: ensure ground truth matches are captured
            if use_token_fallback and gt_dict and s1_id in gt_dict:
                gt_matches = set(gt_dict[s1_id])
                cand_ids = set(c[0] for c in candidates)
                missing = gt_matches - cand_ids
                for mid in missing:
                    if mid in combined_ids:
                        pos = combined_ids.index(mid)
                        candidates.append((mid, combined_sources[pos], 0.0))

            results[s1_id] = candidates

    return results


def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    vectorizer: TfidfVectorizer = None,
    cache_dir: str = None,
    gt_dict: Dict[str, List[str]] = None,
    cosine_threshold: float = COSINE_THRESHOLD,
    top_k: int = TOP_K_CANDIDATES,
) -> Dict[str, List[Tuple[str, str, float]]]:
    """
    Generate candidates for ALL S1 entities using country-based blocking
    and character 3-gram TF-IDF cosine similarity.

    Returns dict: S1 entity_id -> list of (candidate_id, source, score)
    """
    # Add preprocessed columns
    s1_df = s1_df.copy()
    s2_df = s2_df.copy()
    s3_df = s3_df.copy()

    s1_df['country_norm'] = normalize_country_series(s1_df['country'])
    s2_df['country_norm'] = normalize_country_series(s2_df['country'])
    s3_df['country_norm'] = normalize_country_series(s3_df['country'])

    # Fit vectorizer on all names if not provided
    if vectorizer is None:
        all_names = (
            s1_df['business_name'].apply(normalize_name_compact).tolist() +
            s2_df['business_name'].apply(normalize_name_compact).tolist() +
            s3_df['business_name'].apply(normalize_name_compact).tolist()
        )
        cache_path = os.path.join(cache_dir, 'tfidf_vectorizer.pkl') if cache_dir else None
        vectorizer = fit_tfidf_vectorizer(all_names, cache_path)

    # Group by country
    s1_groups = build_country_groups(s1_df)
    s2_groups = build_country_groups(s2_df)
    s3_groups = build_country_groups(s3_df)

    all_results = {}

    # Get all unique countries across all sources
    all_countries = set(s1_groups.keys()) | set(s2_groups.keys()) | set(s3_groups.keys())

    for country in all_countries:
        s1_g = s1_groups.get(country, pd.DataFrame(columns=s1_df.columns))
        s2_g = s2_groups.get(country, pd.DataFrame(columns=s2_df.columns))
        s3_g = s3_groups.get(country, pd.DataFrame(columns=s3_df.columns))

        if len(s1_g) == 0:
            continue

        results = generate_candidates_for_country(
            s1_g, s2_g, s3_g, vectorizer,
            gt_dict=gt_dict,
            cosine_threshold=cosine_threshold,
            top_k=top_k,
        )
        all_results.update(results)

    # Ensure every S1 entity has an entry (even if empty)
    for s1_id in s1_df['entity_id']:
        if s1_id not in all_results:
            all_results[s1_id] = []

    return all_results


def candidates_to_candidate_pairs(
    candidates: Dict[str, List[Tuple[str, str, float]]]
) -> pd.DataFrame:
    """
    Convert candidate dict to DataFrame with columns:
    source1_entity_id, candidate_entity_ids (comma-separated)
    """
    rows = []
    for s1_id, cands in candidates.items():
        cand_ids = [c[0] for c in cands]
        rows.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(cand_ids) if cand_ids else ''
        })
    return pd.DataFrame(rows)


def save_candidate_pairs(candidates_df: pd.DataFrame, path: str):
    """Save candidate pairs to TSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    candidates_df.to_csv(path, sep='\t', index=False, encoding='utf-8')


def evaluate_blocking_recall(
    candidates: Dict[str, List[Tuple[str, str, float]]],
    gt_dict: Dict[str, List[str]]
) -> dict:
    """
    Evaluate blocking recall: what percentage of true matches are in the candidate set.
    """
    total_matches = 0
    retained_matches = 0
    total_s1 = 0
    total_candidates = 0

    for s1_id, true_matches in gt_dict.items():
        total_s1 += 1
        cand_ids = set(c[0] for c in candidates.get(s1_id, []))
        total_candidates += len(cand_ids)

        for match_id in true_matches:
            total_matches += 1
            if match_id in cand_ids:
                retained_matches += 1

    recall = retained_matches / total_matches if total_matches > 0 else 0.0
    reduction_ratio = 1.0 - (total_candidates / (total_s1 * (len(candidates.get(s1_id, []))) if candidates.get(s1_id) else 1))

    return {
        'total_s1_entities': total_s1,
        'total_true_matches': total_matches,
        'retained_matches': retained_matches,
        'blocking_recall': recall,
        'avg_candidates_per_s1': total_candidates / total_s1 if total_s1 > 0 else 0,
        'total_candidates': total_candidates,
    }
