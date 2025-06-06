import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def get_array(file, cleanup: bool = True):
    with open(file, "r") as f:
        data = f.read()
    if cleanup:
        os.remove(file)
    return np.array(list(map(float, data.split())))

def get_t(n, t = None):
    if t is not None:
        t = get_array(t)
    else:
        t = np.arange(n)
    return t

def plot_lines(files):
    if isinstance(files, str):
        files = [files]
    plt.figure(figsize=(15, 8))

    for line in files:
        data = get_array(line)
        plt.plot(data)
        

def plot_errorbar(mean, std, t=None):
    m = get_array(mean)
    s = get_array(std)
    t = get_t(len(m), t)
    plt.figure(figsize=(10, 8))
    plt.errorbar(t, m, yerr=s, capsize=5)


def plot_bar(h, t=None):
    m = get_array(h)
    t = get_t(len(m), t)
    plt.figure(figsize=(15, 8))
    dt = t[1] - t[0]
    plt.bar(t, m, width=dt*1.5)
  
            
def plot_colormesh(dt, *files):
    *files, cf_file = files
    cfs = get_array(cf_file)
    dt = float(dt)
    data = np.vstack([get_array(f) for f in files])
    t = np.arange(data[0].shape[0]) * dt
    plt.figure(figsize=(9, 4))
    plt.pcolor(t, cfs / 1e3, data, cmap="viridis", vmin=0, vmax=data.max())
    plt.yscale("log")
    plt.colorbar()
    

def plot_specgram(dt, *files):
    *files, cf_file = files
    cfs = get_array(cf_file)
    dt = float(dt)
    data = np.vstack([get_array(f) for f in files])
    t = np.arange(data[0].shape[0]) * dt
    plt.figure(figsize=(15, 8))
    plt.pcolormesh(t, cfs, np.log10(data), cmap="nipy_spectral", vmin=0, vmax=data.max())
    plt.colorbar()
    
    

if __name__ == "__main__":
    ptype = sys.argv[1]
    title = sys.argv[2]
    xlabel = sys.argv[3]
    ylabel = sys.argv[4]
    
    if ptype == "bar":
        plot_bar(*sys.argv[5:])

    if ptype == "line":
        plot_lines(sys.argv[5:])

    elif ptype == "errorbar":
        plot_errorbar(*sys.argv[5:])
        
    elif ptype == "colormesh":
        plot_colormesh(*sys.argv[5:])
        
    elif ptype == "specgram":
        plot_specgram(*sys.argv[5:])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
    
        