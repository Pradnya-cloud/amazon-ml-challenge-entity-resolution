"""
Quick end-to-end test script to verify the pipeline works.
Loads full S1+GT (for ID alignment), samples S2/S3 to manageable size.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, 'code', 'business_entity_resolution', 'src'))


def main():
    from data_loader import DATASET_DIR
    from data_loader import ground_truth_to_dict
    from preprocess import normalize_name_compact, normalize_country
    from blocking import fit_tfidf_vectorizer, generate_all_candidates, evaluate_blocking_recall
    from feature_extraction import extract_features_pair, FEATURE_COLUMNS
    from train_matching import compute_macro_f05

    # Load full S1 and GT for proper ID alignment
    nrows = 50000
    print(f"Loading {nrows} rows from each source...")
    t = time.time()
    s1 = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_source1.tsv'),
                     sep='\t', encoding='utf-8', dtype=str, keep_default_na=False, nrows=nrows)
    s2 = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_source2.tsv'),
                     sep='\t', encoding='utf-8', dtype=str, keep_default_na=False, nrows=nrows)
    s3 = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_source3.tsv'),
                     sep='\t', encoding='utf-8', dtype=str, keep_default_na=False, nrows=nrows)
    gt = pd.read_csv(os.path.join(DATASET_DIR, 'train', 'train_ground_truth.tsv'),
                     sep='\t', encoding='utf-8', dtype=str, keep_default_na=False, nrows=nrows)
    print(f"  Done in {time.time()-t:.1f}s")
    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, GT: {len(gt)}")

    # Build GT dict
    gt_dict = ground_truth_to_dict(gt)

    # Find S1 IDs that exist in both S1 and GT (ID alignment)
    s1_valid_ids = set(s1['entity_id'])
    gt_valid_ids = set(gt_dict.keys())
    aligned_ids = sorted(s1_valid_ids & gt_valid_ids)
    print(f"\nAligned S1 IDs (S1 & GT): {len(aligned_ids)}")

    if len(aligned_ids) < 100:
        print("Too few aligned IDs for testing. Try loading more rows.")
        print("S1 suffix range:", list(s1_ids)[:5] if (s1_ids := list(s1_valid_ids)) else "empty")
        print("GT suffix range:", list(gt_ids)[:5] if (gt_ids := list(gt_valid_ids)) else "empty")
        return

    # Build a subset S1 with aligned IDs
    s1_subset = s1[s1['entity_id'].isin(aligned_ids)].reset_index(drop=True)
    gt_subset = {k: v for k, v in gt_dict.items() if k in set(s1_subset['entity_id'])}
    print(f"S1 subset: {len(s1_subset)} entities with GT")

    # Also ensure S2/S3 IDs in GT exist in our S2/S3 samples
    all_matched_ids = set()
    for matches in gt_subset.values():
        all_matched_ids.update(matches)
    s2_ids = set(s2['entity_id'])
    s3_ids = set(s3['entity_id'])
    matched_s2 = all_matched_ids & s2_ids
    matched_s3 = all_matched_ids & s3_ids
    print(f"Matched IDs in S2 sample: {len(matched_s2)}, in S3 sample: {len(matched_s3)}")
    print(f"Total matched IDs: {len(all_matched_ids)}")

    # Split train/val
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(s1_subset))
    n_val = min(500, len(s1_subset) // 4)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_s1 = s1_subset.iloc[train_idx].reset_index(drop=True)
    val_s1 = s1_subset.iloc[val_idx].reset_index(drop=True)

    train_gt = {k: v for k, v in gt_subset.items() if k in set(train_s1['entity_id'])}
    val_gt = {k: v for k, v in gt_subset.items() if k in set(val_s1['entity_id'])}

    print(f"\nTrain S1: {len(train_s1)} ({sum(1 for v in train_gt.values() if v)} with matches)")
    print(f"Val S1: {len(val_s1)} ({sum(1 for v in val_gt.values() if v)} with matches)")

    # Fit TF-IDF vectorizer
    print("\nFitting TF-IDF vectorizer...")
    t = time.time()
    all_names = (
        train_s1['business_name'].apply(normalize_name_compact).tolist() +
        s2['business_name'].apply(normalize_name_compact).tolist() +
        s3['business_name'].apply(normalize_name_compact).tolist()
    )
    vectorizer = fit_tfidf_vectorizer(all_names)
    print(f"  Done in {time.time()-t:.1f}s, vocab: {len(vectorizer.vocabulary_)}")

    # Generate candidates
    print("\nGenerating candidates (train)...")
    t = time.time()
    candidates = generate_all_candidates(
        train_s1, s2, s3,
        vectorizer=vectorizer,
        gt_dict=train_gt,
        cosine_threshold=0.05,
        top_k=100,
    )
    print(f"  Done in {time.time()-t:.1f}s")

    blocking_stats = evaluate_blocking_recall(candidates, train_gt)
    print(f"Blocking stats: {blocking_stats}")

    # Extract features
    print("\nExtracting features (train)...")
    t = time.time()
    s2_lookup = s2.set_index('entity_id')
    s3_lookup = s3.set_index('entity_id')

    feature_rows = []
    labels = []

    for s1_id, cand_list in candidates.items():
        if len(cand_list) == 0:
            continue
        s1_rows = train_s1[train_s1['entity_id'] == s1_id]
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
            feats['s1_id'] = s1_id
            feats['candidate_id'] = cand_id
            feats['source'] = 0.0 if source == 'S2' else 1.0
            feats['blocking_score'] = float(score)
            feature_rows.append(feats)

            true_matches = train_gt.get(s1_id, [])
            labels.append(1 if cand_id in true_matches else 0)

    features_df = pd.DataFrame(feature_rows)
    labels = np.array(labels, dtype=np.int32)
    print(f"  Done in {time.time()-t:.1f}s")
    n_pos = labels.sum()
    print(f"  Feature rows: {len(features_df)}, positive: {n_pos}, negative: {len(labels)-n_pos}")

    if n_pos == 0:
        print("  WARNING: No positive labels!")
        return

    # Train LightGBM
    print("\nTraining LightGBM model...")
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    feature_cols = FEATURE_COLUMNS + ['blocking_score', 'source']
    X = features_df[feature_cols].copy()
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1) * 0.5
    model = lgb.LGBMClassifier(
        n_estimators=50, max_depth=6, num_leaves=31,
        learning_rate=0.1, scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate threshold
    probs = model.predict_proba(X_val)[:, 1]
    prob_df = pd.DataFrame({
        's1_id': features_df['s1_id'].iloc[X_val.index].values,
        'candidate_id': features_df['candidate_id'].iloc[X_val.index].values,
        'probability': probs,
    })

    best_f05 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.05, 0.95, 19):
        predictions = {}
        above = prob_df[prob_df['probability'] >= thresh]
        for s1_id, group in above.groupby('s1_id'):
            predictions[s1_id] = list(group['candidate_id'])
        for s1_id in val_gt:
            if s1_id not in predictions:
                predictions[s1_id] = []
        stats = compute_macro_f05(val_gt, predictions)
        if stats['macro_f05'] > best_f05:
            best_f05 = stats['macro_f05']
            best_thresh = thresh

    print(f"\nBest threshold F0.5 on train-split val: {best_f05:.4f} at {best_thresh:.3f}")

    # Now evaluate on the separate val set
    print("\nGenerating candidates (val)...")
    val_candidates = generate_all_candidates(
        val_s1, s2, s3,
        vectorizer=vectorizer,
        gt_dict=val_gt,
        cosine_threshold=0.05,
        top_k=100,
    )

    val_feature_rows = []
    for s1_id, cand_list in val_candidates.items():
        if len(cand_list) == 0:
            continue
        s1_rows = val_s1[val_s1['entity_id'] == s1_id]
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
            feats['s1_id'] = s1_id
            feats['candidate_id'] = cand_id
            feats['source'] = 0.0 if source == 'S2' else 1.0
            feats['blocking_score'] = float(score)
            val_feature_rows.append(feats)

    val_features_df = pd.DataFrame(val_feature_rows)
    print(f"  Val feature rows: {len(val_features_df)}")

    if len(val_features_df) > 0:
        val_probs = model.predict_proba(val_features_df[feature_cols])[:, 1]
        val_pred_df = pd.DataFrame({
            's1_id': val_features_df['s1_id'].values,
            'candidate_id': val_features_df['candidate_id'].values,
            'probability': val_probs,
        })

        predictions = {}
        above = val_pred_df[val_pred_df['probability'] >= best_thresh]
        for s1_id, group in above.groupby('s1_id'):
            predictions[s1_id] = list(group['candidate_id'])
        for s1_id in val_gt:
            if s1_id not in predictions:
                predictions[s1_id] = []

        stats = compute_macro_f05(val_gt, predictions)
        print(f"\nValidation F0.5: {stats['macro_f05']:.4f}")
        print(f"  Precision: {stats['macro_precision']:.4f}")
        print(f"  Recall: {stats['macro_recall']:.4f}")
        print(f"  S1 entities: {stats['n_entities']}")
        print(f"  Matched (non-singleton) F0.5: {stats['matched_f05']:.4f}")
        print(f"  Singleton F0.5: {stats['singleton_f05']:.4f}")

    print("\nQuick test PASSED!")


if __name__ == '__main__':
    main()
