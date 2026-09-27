"""
Ultra-fast full-test prediction using pandas merge for exact name matching.
Completes in ~5-10 minutes for full 1.7M test set.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
sys.path.insert(0, _THIS_DIR)

from data_loader import DATASET_DIR
from preprocess import normalize_country_series

OUTPUT_DIR = os.path.join(_BASE_DIR, 'output')

STOPWORDS = {'the', 'and', 'for', 'ltd', 'llc', 'inc', 'co', 'corp', 'group', 'services',
             'service', 'private', 'pvt', 'limited', 'trading', 'company', 'enterprises'}


def normalize_name_fast(name_series: pd.Series) -> pd.Series:
    names = name_series.str.lower().str.strip()
    names = names.str.replace(r'[^\w\s]', ' ', regex=True)
    names = names.str.replace(r'\s+', ' ', regex=True)
    return names.apply(lambda s: ' '.join(sorted(set(t for t in s.split() if len(t) >= 2 and t not in STOPWORDS))))


def main():
    start = time.time()
    print("Loading all test data...")

    s1 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source1.tsv'),
                     sep='\t', encoding='utf-8', dtype=str, keep_default_na=False)
    s2 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source2.tsv'),
                     sep='\t', encoding='utf-8', dtype=str, keep_default_na=False)
    s3 = pd.read_csv(os.path.join(DATASET_DIR, 'test', 'test_source3.tsv'),
                     sep='\t', encoding='utf-8', dtype=str, keep_default_na=False)

    for df in [s1, s2, s3]:
        for col in ['business_name', 'business_address', 'country']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)

    print(f"Loaded: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}, Time: {time.time()-start:.1f}s")

    s1['country_norm'] = normalize_country_series(s1['country'])
    s2['country_norm'] = normalize_country_series(s2['country'])
    s3['country_norm'] = normalize_country_series(s3['country'])

    print("Normalizing names...")
    s1['name_key'] = normalize_name_fast(s1['business_name'])
    s2['name_key'] = normalize_name_fast(s2['business_name'])
    s3['name_key'] = normalize_name_fast(s3['business_name'])
    print(f"  Done, Time: {time.time()-start:.1f}s")

    # Remove empty name_keys to reduce merge size
    s1_nn = s1[s1['name_key'].str.len() > 0].copy()
    s2_nn = s2[s2['name_key'].str.len() > 0].copy()
    s3_nn = s3[s3['name_key'].str.len() > 0].copy()

    print(f"Non-empty names: S1={len(s1_nn)}, S2={len(s2_nn)}, S3={len(s3_nn)}")

    # Merge S1 with S2 and S3 on name_key + country_norm
    print("Merging with S2...")
    s2_matches = s1_nn[['entity_id', 'name_key', 'country_norm']].merge(
        s2_nn[['entity_id', 'name_key', 'country_norm']],
        on=['name_key', 'country_norm'], suffixes=('', '_cand')
    )
    s2_matches['candidate_id'] = s2_matches['entity_id_cand'].apply(lambda x: f'S2-{x}')
    print(f"  S2 matches: {len(s2_matches)}, Time: {time.time()-start:.1f}s")

    print("Merging with S3...")
    s3_matches = s1_nn[['entity_id', 'name_key', 'country_norm']].merge(
        s3_nn[['entity_id', 'name_key', 'country_norm']],
        on=['name_key', 'country_norm'], suffixes=('', '_cand')
    )
    s3_matches['candidate_id'] = s3_matches['entity_id_cand'].apply(lambda x: f'S3-{x}')
    print(f"  S3 matches: {len(s3_matches)}, Time: {time.time()-start:.1f}s")

    # Combine matches
    s2_matches = s2_matches[['entity_id', 'candidate_id']]
    s3_matches = s3_matches[['entity_id', 'candidate_id']]
    all_matches = pd.concat([s2_matches, s3_matches], ignore_index=True)

    print(f"Total exact matches: {len(all_matches)}, Time: {time.time()-start:.1f}s")

    # Group by S1 entity_id
    print("Grouping matches...")
    grouped = all_matches.groupby('entity_id')['candidate_id'].apply(list).to_dict()

    print(f"S1 entities with matches: {len(grouped)}, Time: {time.time()-start:.1f}s")

    # Write output
    print("Writing output files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')

    cand_f = open(cand_path, 'w', encoding='utf-8')
    match_f = open(match_path, 'w', encoding='utf-8')
    cand_f.write('source1_entity_id\tcandidate_entity_ids\n')
    match_f.write('source1_entity_id\tmatched_entity_ids\n')

    total_matches = 0
    for _, s1_row in s1.iterrows():
        s1_id = s1_row['entity_id']
        cands = grouped.get(s1_id, [])
        cand_ids = cands[:10]
        match_ids = cands[:5]
        cand_f.write(f'{s1_id}\t{",".join(cand_ids) if cand_ids else ""}\n')
        match_f.write(f'{s1_id}\t{",".join(match_ids) if match_ids else ""}\n')
        total_matches += len(match_ids)

    cand_f.close()
    match_f.close()

    elapsed = time.time() - start
    print(f"\nDone! Total S1: {len(s1)}, Matches: {total_matches}, Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
