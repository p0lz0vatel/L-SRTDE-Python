import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PATH = "Results/"
NFunc = 2
NRuns = 31
AllRes = np.zeros((NFunc, NRuns, 1001))
AllRes[1] = np.loadtxt(PATH + f"L-SRTDE_GNBG_F{2}_D30.txt")
AllRes[0] = np.loadtxt(PATH + f"L-SRTDE_GNBG_F{1}_D30_python.txt")

fig = plt.figure(figsize=(15, 9), constrained_layout=True)
gs = fig.add_gridspec(6, 4)
ax = []
for func in range(NFunc):
    ax.append(fig.add_subplot(gs[func]))
for func in range(NFunc):
    for r in range(NRuns):
        ax[func].plot(AllRes[func, r, :-1])
    ax[func].set_title(f"F{func + 1}")
    ax[func].set_yscale("log")
fig.savefig(PATH + "L-SRTDE_GNBG_24_GRAPHS.pdf")
fig.savefig(PATH + "L-SRTDE_GNBG_24_GRAPHS.png")
plt.show()
