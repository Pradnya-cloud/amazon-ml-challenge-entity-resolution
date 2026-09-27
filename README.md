# Amazon ML Challenge - Business Entity Resolution

## Overview
This project solves the Business Entity Resolution task from the Amazon ML Challenge.
The goal is to match S1 entities (source1_entity_id) to their matches in S2 and S3
candidate entities based on business names, addresses, and countries.

## Project Structure
```
amazon_ml_challenge/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── data_loader.py       # TSV loading and preprocessing
│       │   ├── preprocess.py        # Name/address/country normalization
│       │   ├── blocking.py          # TF-IDF 3-gram cosine similarity blocking
│       │   ├── feature_extraction.py # 37+ features per candidate pair
│       │   ├── train_matching.py    # LightGBM model training
│       │   ├── predict.py           # Prediction pipeline (model-based)
│       │   ├── fast_predict.py      # Fast prediction with smaller data
│       │   └── ultra_fast_predict.py # Full test set prediction (8 min)
│       └── requirements.txt
├── dataset/
│   ├── train/           # Training TSV files (gitignored)
│   └── test/            # Test TSV files (gitignored)
├── scratch/
│   ├── models/          # Trained model artifacts (gitignored)
│   ├── features/
│   └── logs/
├── output/
│   ├── candidate_pairs.tsv      # Candidate pairs output
│   └── matching_results.tsv     # Final matching results
├── utils/
│   └── validate_submission.py   # Output format validation
├── submission.zip       # Packaged submission (gitignored)
├── package_submission.py
└── README.md
```

## Quick Start

### Train Model
```bash
cd code/business_entity_resolution/src
py -3.14 -u train_matching.py
```

### Run Prediction (Full Test Set)
```bash
cd code/business_entity_resolution/src
py -3.14 -u ultra_fast_predict.py
```

### Validate Output
```bash
cd utils
py -3.14 -u validate_submission.py
```

### Package Submission
```bash
py -3.14 -u package_submission.py
```

## Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
lightgbm>=4.0.0
rapidfuzz>=3.0.0
python-Levenshtein>=0.23.0
```

Install with:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## Approach
1. **Blocking**: Character 3-gram TF-IDF cosine similarity with country pre-filtering
2. **Feature Extraction**: 37 features including name/address similarity metrics
3. **Model**: LightGBM classifier with class imbalance handling
4. **Prediction**: Exact name matching with normalization + model scoring

## Team
- [Add team members here]
