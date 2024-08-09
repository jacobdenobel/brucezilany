import os
import pickle
from dataclasses import dataclass, field
import argparse

import numpy as np
import matplotlib.pyplot as plt

import bruce


FIGURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/data/figures")

def plot_neurogram(t, y, data, save_to: str = None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.pcolor(
        t, y, data, cmap="viridis", vmin=0, vmax=data.max()
    )
    ax.set_yscale("log")
    ax.set_ylabel("frequency")
    ax.set_xlabel("time [s]")
    ax.colorbar()
    if save_to is not None:
        plt.savefig(save_to)
    return as
    
    
class Neurogram:
    f: np.array 
    t: np.array = field(repr=None)
    data: np.array = field(repr=None)
    name: str = None
    
    def save(self, path: str):
        with open(path, "wb") as f:    
            pickle.dump(self, f) 
        

if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)

    parser = argparse.ArgumentParser("Create a neurogram for a given WAV file at path")
    parser.add_argument("path", type=str)
    parser.add_argument("--n_cf", default=40, type=int)
    parser.add_argument("--seed", default=40, type=int)
    parser.add_argument("--n_rep", default=1, type=int)
    parser.add_argument("--bin_width", default=5e-4, type=float)
    parser.add_argument("--plot", action="store_true")
    
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(args.path + " not found")

    bruce.set_seed(args.seed)
    np.random.seed(args.seed)

    stim = bruce.stimulus.from_file(args.path, True)
    ng = bruce.Neurogram(args.n_cf)
    ng.bin_width = args.bin_width
    ng.create(stim, args.n_rep)

    binned_output = ng.get_unfiltered_output()
    
    y = ng.get_cfs()
    mean_timing, dt_mean = ng.get_mean_timing()
    
    if args.plot:
        plot_neurogram(
            np.arange(binned_output.shape[1]) * args.bin_width, y, binned_output, 
            f"{FIGURES}/{os.path.basename(args.path)}_binned.png"
        )
        plot_neurogram(
            np.arange(mean_timing.shape[1]) * dt_mean, y, mean_timing, 
            f"{FIGURES}/{os.path.basename(args.path)}_mean.png"
        )
        