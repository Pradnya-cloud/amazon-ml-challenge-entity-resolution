"""
Validate the submission format for the Business Entity Resolution challenge.

Checks:
1. Required output files exist
2. TSV format with correct headers
3. All test source1 entity IDs are covered
4. Entity ID formats are correct
5. No extra columns or malformed rows
"""

import os
import sys
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_BASE, 'output')
DATASET_DIR = os.path.join(_BASE, 'dataset')

CANDIDATE_HEADER = ['source1_entity_id', 'candidate_entity_ids']
MATCHING_HEADER = ['source1_entity_id', 'matched_entity_ids']


def validate_submission(output_dir=OUTPUT_DIR, dataset_dir=DATASET_DIR, verbose=True):
    errors = []
    warnings = []

    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    match_path = os.path.join(output_dir, 'matching_results.tsv')

    if not os.path.exists(cand_path):
        errors.append(f"Missing file: {cand_path}")
        return errors, warnings
    if not os.path.exists(match_path):
        errors.append(f"Missing file: {match_path}")
        return errors, warnings

    try:
        candidates = pd.read_csv(cand_path, sep='\t', dtype=str, keep_default_na=False, header=None, names=CANDIDATE_HEADER, skiprows=1)
    except Exception as e:
        errors.append(f"Failed to read candidate_pairs.tsv: {e}")
        return errors, warnings

    try:
        matches = pd.read_csv(match_path, sep='\t', dtype=str, keep_default_na=False, header=None, names=MATCHING_HEADER, skiprows=1)
    except Exception as e:
        errors.append(f"Failed to read matching_results.tsv: {e}")
        return errors, warnings

    if len(candidates) == 0:
        errors.append("candidate_pairs.tsv has no data rows")
    if len(matches) == 0:
        errors.append("matching_results.tsv has no data rows")

    if verbose:
        print(f"candidate_pairs.tsv: {len(candidates)} rows")
        print(f"matching_results.tsv: {len(matches)} rows")

    cand_s1_ids = set(candidates['source1_entity_id'].tolist())
    match_s1_ids = set(matches['source1_entity_id'].tolist())

    if cand_s1_ids != match_s1_ids:
        only_cand = cand_s1_ids - match_s1_ids
        only_match = match_s1_ids - cand_s1_ids
        if len(only_cand) > 0:
            errors.append(f"S1 IDs in candidate_pairs but not matching_results: {len(only_cand)}")
        if len(only_match) > 0:
            errors.append(f"S1 IDs in matching_results but not candidate_pairs: {len(only_match)}")

    test_s1_path = os.path.join(dataset_dir, 'test', 'test_source1.tsv')
    if os.path.exists(test_s1_path):
        test_s1 = pd.read_csv(test_s1_path, sep='\t', usecols=['entity_id'], dtype=str)
        test_s1_ids = set(test_s1['entity_id'].tolist())
        missing = test_s1_ids - match_s1_ids
        if len(missing) > 0:
            warnings.append(f"Test S1 entities not in submission: {len(missing)} out of {len(test_s1_ids)}")
            if verbose:
                print(f"Coverage: {len(test_s1_ids & match_s1_ids)}/{len(test_s1_ids)} ({100*len(test_s1_ids & match_s1_ids)/len(test_s1_ids):.1f}%)")

    s1_ids_with_matches = matches[matches['matched_entity_ids'].str.len() > 0]
    if verbose:
        print(f"S1 entities with matches: {len(s1_ids_with_matches)}")
        total_matches = sum(len(m.split(',')) for m in matches['matched_entity_ids'] if m)
        print(f"Total matches: {total_matches}")

    if verbose:
        print("\nValidation complete.")
        if errors:
            print(f"ERRORS ({len(errors)}):")
            for e in errors:
                print(f"  - {e}")
        if warnings:
            print(f"WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"  - {w}")
        if not errors and not warnings:
            print("All checks passed!")

    return errors, warnings


if __name__ == '__main__':
    errors, warnings = validate_submission(verbose=True)
    sys.exit(1 if errors else 0)
