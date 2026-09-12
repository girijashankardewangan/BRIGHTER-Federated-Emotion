#!/usr/bin/env python3
import os, sys, json, time
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_goemotions, train_centralized, train_federated, METHODS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SEEDS = [0, 1, 2, 3, 4]
PROGRESS_FILE = "goemotions_progress.json"
RESULTS_FILE = "goemotions_results.csv"

print("Loading GoEmotions...")
go_train, go_val, go_test = load_goemotions()

if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
else:
    progress = {"completed": []}
completed = set(tuple(x) for x in progress["completed"])

if os.path.exists(RESULTS_FILE):
    results_df = pd.read_csv(RESULTS_FILE)
else:
    results_df = pd.DataFrame(columns=["dataset","method","seed","macro_f1","micro_f1","exact_match"])

for seed in SEEDS:
    for method in METHODS:
        key = ("GoEmotions", method, seed)
        if key in completed:
            print(f"Skipping {key}")
            continue
        print(f"\n{'='*60}\nRunning GoEmotions | {method} | seed {seed}\n{'='*60}")
        start = time.time()
        try:
            if method == "C1":
                macro, micro, exact = train_centralized(go_train, go_val, go_test, seed)
            else:
                macro, micro, exact = train_federated(go_train, go_val, go_test, seed, method)
        except Exception as e:
            print(f"Error: {e}"); import traceback; traceback.print_exc(); continue
        print(f"Done in {(time.time()-start)/60:.2f} min | M={macro:.4f} m={micro:.4f} E={exact:.4f}")
        new_row = {"dataset":"GoEmotions","method":method,"seed":seed,
                   "macro_f1":float(macro),"micro_f1":float(micro),"exact_match":float(exact)}
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        results_df.to_csv(RESULTS_FILE, index=False)
        completed.add(key)
        progress["completed"] = [list(x) for x in completed]
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
print("\n🎉 Done!")
