import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import bruce

FIGURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/data/figures")

def plot_neurogram(t, y, data, save_to: str = None):
    plt.figure(figsize=(9, 4))
    plt.pcolor(
        t, y, data, cmap="viridis", vmin=0, vmax=data.max()
    )
    plt.yscale("log")
    plt.ylabel("frequency")
    plt.xlabel("time [s]")
    plt.colorbar()
    if save_to is not None:
        plt.savefig(save_to)
    

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
        