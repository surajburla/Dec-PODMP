import pandas as pd
import glob
import numpy as np
import os

files = glob.glob("results/*/log.csv")

if not files:
    raise FileNotFoundError("Missing log.csv files in results/ subdirectories.")

for file_path in files:
    policy_name = os.path.basename(os.path.dirname(file_path))
    df = pd.read_csv(file_path)
    
    if 'ret' not in df.columns:
        print(f"[{policy_name}] FAILED: 'ret' column missing.")
        continue
        
    returns = df['ret'].values
    p05 = np.percentile(returns, 5)
    sub_120_count = np.sum(returns < 120)
    
    print(f"Policy: {policy_name}")
    print(f"  5th Percentile Return: {p05:.2f}")
    print(f"  Sub-120 Failures:      {sub_120_count}\n")