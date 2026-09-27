"""
Feature extraction module for Business Entity Resolution.

Computes similarity and matching features for S1 <-> S2/S3 candidate pairs.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from rapidfuzz import fuzz, distance as rdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy import sparse
from preprocess import (
    normalize_name, normalize_name_compact, normalize_address, normalize_country,
    extract_tokens, extract_char_ngrams, extract_postal_code, extract_house_number
)

# Cache for TF-IDF vectorizers (per column type)
_tfidf_cache = {}

# Country-pair match frequency table (populated during training)
_country_pair_freq = {}
_COUNTRY_PAIR_DEFAULT = 0.1


def set_country_pair_freq(freq: Dict[Tuple[str, str], float]):
    """Set the country-pair match frequency table from training data."""
    global _country_pair_freq
    _country_pair_freq = freq


def get_country_pair_freq(s1_country: str, s2_country: str) -> float:
    """Get the match frequency for a country pair."""
    key = (s1_country, s2_country)
    return _country_pair_freq.get(key, _COUNTRY_PAIR_DEFAULT)


def _get_tfidf_matrix(texts: List[str], cache_key: str, ngram_range=(1, 2)):
    """Get or create TF-IDF matrix for a list of texts."""
    if cache_key not in _tfidf_cache:
        vec = TfidfVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            lowercase=True,
            token_pattern=r'\b[a-z0-9]+\b',
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            max_features=50000,
        )
        matrix = vec.fit_transform(texts)
        _tfidf_cache[cache_key] = (vec, matrix)
    return _tfidf_cache[cache_key]


def _char_tfidf_matrix(texts: List[str], cache_key: str, n=3):
    """Get or create character n-gram TF-IDF matrix."""
    if cache_key not in _tfidf_cache:
        vec = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(n, n),
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            max_features=50000,
        )
        matrix = vec.fit_transform(texts)
        _tfidf_cache[cache_key] = (vec, matrix)
    return _tfidf_cache[cache_key]


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def overlap_coefficient(set1: set, set2: set) -> float:
    """Compute overlap coefficient between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    return intersection / min(len(set1), len(set2))


def cosine_similarity_tfidf(vec1, vec2) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    if vec1 is None or vec2 is None:
        return 0.0
    # Handle sparse vectors
    if sparse.issparse(vec1):
        v1 = vec1.toarray().flatten()
    else:
        v1 = np.asarray(vec1).flatten()
    if sparse.issparse(vec2):
        v2 = vec2.toarray().flatten()
    else:
        v2 = np.asarray(vec2).flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute Levenshtein similarity ratio using rapidfuzz."""
    if not s1 or not s2:
        return 0.0
    return fuzz.ratio(s1, s2) / 100.0


def partial_ratio(s1: str, s2: str) -> float:
    """Compute partial ratio similarity."""
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


def token_sort_ratio(s1: str, s2: str) -> float:
    """Compute token sort ratio similarity."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def token_set_ratio(s1: str, s2: str) -> float:
    """Compute token set ratio similarity."""
    if not s1 or not s2:
        return 0.0
    return fuzz.token_set_ratio(s1, s2) / 100.0


def extract_features_pair(
    s1_name: str, s1_addr: str, s1_country: str,
    s2_name: str, s2_addr: str, s2_country: str,
) -> Dict[str, float]:
    """
    Extract features for a single S1 <-> S2/S3 pair.

    Returns a dictionary of feature name -> value.
    """
    features = {}

    # Normalize inputs
    s1_name_norm = normalize_name_compact(s1_name)
    s2_name_norm = normalize_name_compact(s2_name)
    s1_addr_norm = normalize_address(s1_addr)
    s2_addr_norm = normalize_address(s2_addr)
    s1_country_norm = normalize_country(s1_country)
    s2_country_norm = normalize_country(s2_country)

    # --- Name features ---
    # Exact normalized match
    features['name_exact_match'] = 1.0 if s1_name_norm == s2_name_norm and s1_name_norm != '' else 0.0

    # Character 3-gram Jaccard
    s1_3grams = extract_char_ngrams(s1_name_norm, 3)
    s2_3grams = extract_char_ngrams(s2_name_norm, 3)
    features['name_char3gram_jaccard'] = jaccard_similarity(s1_3grams, s2_3grams)

    # Character 3-gram overlap
    features['name_char3gram_overlap'] = overlap_coefficient(s1_3grams, s2_3grams)

    # Token Jaccard (word-level)
    s1_tokens = extract_tokens(s1_name_norm)
    s2_tokens = extract_tokens(s2_name_norm)
    features['name_token_jaccard'] = jaccard_similarity(s1_tokens, s2_tokens)

    # Token overlap coefficient
    features['name_token_overlap'] = overlap_coefficient(s1_tokens, s2_tokens)

    # Levenshtein / fuzzy ratios
    features['name_levenshtein_ratio'] = levenshtein_ratio(s1_name_norm, s2_name_norm)
    features['name_partial_ratio'] = partial_ratio(s1_name_norm, s2_name_norm)
    features['name_token_sort_ratio'] = token_sort_ratio(s1_name_norm, s2_name_norm)
    features['name_token_set_ratio'] = token_set_ratio(s1_name_norm, s2_name_norm)

    # Length difference ratio
    len1, len2 = len(s1_name_norm), len(s2_name_norm)
    if len1 + len2 > 0:
        features['name_length_diff_ratio'] = abs(len1 - len2) / (len1 + len2)
    else:
        features['name_length_diff_ratio'] = 1.0

    # --- Address features ---
    features['addr_exact_match'] = 1.0 if s1_addr_norm == s2_addr_norm and s1_addr_norm != '' else 0.0
    features['addr_char3gram_jaccard'] = jaccard_similarity(
        extract_char_ngrams(s1_addr_norm, 3), extract_char_ngrams(s2_addr_norm, 3)
    )
    features['addr_char3gram_overlap'] = overlap_coefficient(
        extract_char_ngrams(s1_addr_norm, 3), extract_char_ngrams(s2_addr_norm, 3)
    )

    s1_addr_tokens = set(s1_addr_norm.split()) if s1_addr_norm else set()
    s2_addr_tokens = set(s2_addr_norm.split()) if s2_addr_norm else set()
    features['addr_token_jaccard'] = jaccard_similarity(s1_addr_tokens, s2_addr_tokens)
    features['addr_token_overlap'] = overlap_coefficient(s1_addr_tokens, s2_addr_tokens)
    features['addr_levenshtein_ratio'] = levenshtein_ratio(s1_addr_norm, s2_addr_norm)
    features['addr_partial_ratio'] = partial_ratio(s1_addr_norm, s2_addr_norm)
    features['addr_token_sort_ratio'] = token_sort_ratio(s1_addr_norm, s2_addr_norm)
    features['addr_token_set_ratio'] = token_set_ratio(s1_addr_norm, s2_addr_norm)

    # Postal code match
    s1_pc = extract_postal_code(s1_addr)
    s2_pc = extract_postal_code(s2_addr)
    features['postal_code_match'] = 1.0 if s1_pc == s2_pc and s1_pc != '' else 0.0

    # House number match
    s1_hn = extract_house_number(s1_addr)
    s2_hn = extract_house_number(s2_addr)
    features['house_number_match'] = 1.0 if s1_hn == s2_hn and s1_hn != '' else 0.0

    # Address length diff
    alen1, alen2 = len(s1_addr_norm), len(s2_addr_norm)
    if alen1 + alen2 > 0:
        features['addr_length_diff_ratio'] = abs(alen1 - alen2) / (alen1 + alen2)
    else:
        features['addr_length_diff_ratio'] = 1.0

    # --- Other features ---
    features['country_match'] = 1.0 if s1_country_norm == s2_country_norm and s1_country_norm != '' else 0.0
    features['s1_country_us'] = 1.0 if s1_country_norm == 'US' else 0.0
    features['s1_country_india'] = 1.0 if s1_country_norm == 'India' else 0.0
    features['s1_country_france'] = 1.0 if s1_country_norm == 'France' else 0.0
    features['s2_country_us'] = 1.0 if s2_country_norm == 'US' else 0.0
    features['s2_country_india'] = 1.0 if s2_country_norm == 'India' else 0.0
    features['s2_country_france'] = 1.0 if s2_country_norm == 'France' else 0.0

    # Country-pair match frequency
    features['country_pair_freq'] = get_country_pair_freq(s1_country_norm, s2_country_norm)

    # Combined similarity
    features['combined_name_addr'] = (
        features['name_token_set_ratio'] * 0.6 +
        features['addr_token_set_ratio'] * 0.4
    )

    # Missing value indicators
    features['s1_name_missing'] = 1.0 if not s1_name_norm else 0.0
    features['s2_name_missing'] = 1.0 if not s2_name_norm else 0.0
    features['s1_addr_missing'] = 1.0 if not s1_addr_norm else 0.0
    features['s2_addr_missing'] = 1.0 if not s2_addr_norm else 0.0

    return features


FEATURE_COLUMNS = [
    'name_exact_match',
    'name_char3gram_jaccard',
    'name_char3gram_overlap',
    'name_token_jaccard',
    'name_token_overlap',
    'name_levenshtein_ratio',
    'name_partial_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_length_diff_ratio',
    'addr_exact_match',
    'addr_char3gram_jaccard',
    'addr_char3gram_overlap',
    'addr_token_jaccard',
    'addr_token_overlap',
    'addr_levenshtein_ratio',
    'addr_partial_ratio',
    'addr_token_sort_ratio',
    'addr_token_set_ratio',
    'postal_code_match',
    'house_number_match',
    'addr_length_diff_ratio',
    'country_match',
    's1_country_us', 's1_country_india', 's1_country_france',
    's2_country_us', 's2_country_india', 's2_country_france',
    'country_pair_freq',
    'combined_name_addr',
    's1_name_missing', 's2_name_missing',
    's1_addr_missing', 's2_addr_missing',
]


def extract_features_batch(
    s1_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    gt_dict: Dict[str, List[str]] = None,
    source_label: str = 'S2'
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features for a batch of S1 entities and their candidates.

    Args:
        s1_df: Source 1 entities DataFrame
        candidate_df: DataFrame with columns ['s1_id', 'candidate_id', 'candidate_name', 'candidate_addr', 'candidate_country', 'source']
        gt_dict: ground truth mapping S1 ID -> list of matched IDs
        source_label: 'S2' or 'S3'

    Returns:
        features_df: DataFrame with feature columns
        labels: numpy array of 0/1 labels
    """
    feature_rows = []
    labels = []

    gt_set = gt_dict or {}

    for _, crow in candidate_df.iterrows():
        s1_id = crow['s1_id']
        cand_id = crow['candidate_id']
        s1_row = s1_df[s1_df['entity_id'] == s1_id]
        if len(s1_row) == 0:
            continue
        s1_row = s1_row.iloc[0]

        feats = extract_features_pair(
            str(s1_row['business_name']),
            str(s1_row['business_address']),
            str(s1_row['country']),
            str(crow['candidate_name']),
            str(crow['candidate_addr']),
            str(crow['candidate_country'])
        )
        feats['s1_id'] = s1_id
        feats['candidate_id'] = cand_id
        feature_rows.append(feats)

        # Label: 1 if candidate is in ground truth for this S1
        matches = gt_set.get(s1_id, [])
        labels.append(1 if cand_id in matches else 0)

    features_df = pd.DataFrame(feature_rows)
    labels = np.array(labels, dtype=np.int32)
    return features_df, labels
