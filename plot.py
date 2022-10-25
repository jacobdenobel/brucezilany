import sys
import numpy as np
import matplotlib.pyplot as plt

def get_array(file):
    with open(file, "r") as f:
        data = f.read()
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
  
            

if __name__ == "__main__":
    ptype = sys.argv[1]
    title = sys.argv[2]
    xlabel = sys.argv[3]
    ylabel = sys.argv[4]
    
    if ptype == "bar":
        plot_bar(*sys.argv[5:])

    if ptype == "line":
        plot_lines(sys.argv[5:])

    elif ptype == 'errorbar':
        plot_errorbar(*sys.argv[5:])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
        