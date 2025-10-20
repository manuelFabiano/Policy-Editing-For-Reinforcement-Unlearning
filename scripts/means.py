import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from scipy import stats

# -------- 1. Load JSON file --------
path = Path("/home/microway/ManuelFabianoStuff/code/algos/dqn/spaceinvaders-final-seed-uniform_results.json")         
with path.open() as f:
    data = json.load(f)

# -------- 2. Reorganize by traj_len --------
by_len = {}
for run in data:
    L = run["traj_len"]
    d = by_len.setdefault(L, {
        "all_returns": [],
        "forget_tgt": [],
        "retain_tgt": [],
    })

    
    d["all_returns"].extend(x for x in run["metrics2"]["all_returns"])
    d["forget_tgt"].append(run["metrics1"]["forget_target_mean_distance"])
    d["retain_tgt"].append(run["metrics1"]["retain_target_mean_distance"])

# -------- 3. Calculate statistics --------
rows = []
for L, d in by_len.items():
    returns = np.array(d["all_returns"], dtype=float)

    sem = stats.sem(returns)  # Standard Error of the Mean
    confidence = 0.95

        # Calculate confidence interval
    h = sem * stats.t.ppf((1 + confidence) / 2., len(returns)-1)

    rows.append({
        "traj_len": L,
        "returns_mean":   returns.mean(),
        "returns_median": np.median(returns),
        "returns_ci":    h,  # ddof=1 → sample standard deviation
        "avg_forget_target_mean_distance": np.mean(d["forget_tgt"]),
        "avg_retain_target_mean_distance": np.mean(d["retain_tgt"]),
        "n_returns": len(returns),              # optional: how many observations
        "n_runs":    len(d["forget_tgt"]),      # optional: how many runs
    })

df = pd.DataFrame(rows).sort_values("traj_len")
print(df)


df.to_csv("/home/microway/ManuelFabianoStuff/code/algos/dqn/spaceinvaders_uniform.csv", index=False)