
import sys
sys.path.insert(0, ".")
from train import load_goemotions, GOEMOTIONS_SEEDS, METHODS

print(f"GOEMOTIONS_SEEDS = {GOEMOTIONS_SEEDS}")
print(f"METHODS = {METHODS}")
train, val, test = load_goemotions()
print(f"GoEmotions loaded: {len(train)} train, {len(val)} val, {len(test)} test")
