import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import bruce

if __name__ == "__main__":
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
    t = np.arange(binned_output.shape[1]) * args.bin_width
    
    if args.plot:
        plt.figure(figsize=(9, 4))
        plt.pcolor(t, y / 1e3, binned_output, cmap="viridis", vmin=0, vmax=binned_output.max())
        plt.yscale("log")
        plt.ylabel("frequency")
        plt.xlabel("time [s]")
        plt.colorbar()
        plt.savefig(f"../data/figures/{args.path}_neurogram.png")
        # plt.show()
        
                   


    