"""
Step 1: download the "Mobile Game Retention & Churn Analysis" dataset from
Kaggle (rajatsurana979/mobile-game-retention-and-churn-analysis).

Auth: kagglehub reads credentials from ~/.kaggle/access_token (or the
KAGGLE_API_TOKEN env var) automatically. No token is stored in this repo.
"""
import shutil

import kagglehub

path = kagglehub.dataset_download("rajatsurana979/mobile-game-retention-and-churn-analysis")
print("Downloaded to cache at:", path)

import os

for f in os.listdir(path):
    print(" -", f)
    shutil.copy(os.path.join(path, f), f)

print("\nCopied dataset file(s) into the project folder.")
