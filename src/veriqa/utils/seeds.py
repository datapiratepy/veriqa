"""Deterministic seeding for every stochastic component."""
import os, random
import numpy as np

def set_all_seeds(seed: int) -> None:
    """Seed python, numpy and hash randomisation. Call once at script start."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
