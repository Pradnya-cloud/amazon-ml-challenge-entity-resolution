"""
Package the submission files into a ZIP archive.

Creates submission.zip containing:
- output/candidate_pairs.tsv
- output/matching_results.tsv
"""

import os
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SUBMISSION_PATH = os.path.join(BASE_DIR, 'submission.zip')

FILES_TO_PACKAGE = ['candidate_pairs.tsv', 'matching_results.tsv']


def package_submission(output_dir=OUTPUT_DIR, submission_path=SUBMISSION_PATH):
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    files_found = []
    for fname in FILES_TO_PACKAGE:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            files_found.append(fpath)
        else:
            print(f"WARNING: {fpath} not found!")

    if not files_found:
        print("ERROR: No submission files found!")
        return False

    with zipfile.ZipFile(submission_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fpath in files_found:
            zf.write(fpath, arcname=os.path.basename(fpath))
            print(f"  Added: {os.path.basename(fpath)}")

    print(f"\nSubmission packaged: {submission_path}")
    print(f"Total files: {len(files_found)}")
    return True


if __name__ == '__main__':
    package_submission()
