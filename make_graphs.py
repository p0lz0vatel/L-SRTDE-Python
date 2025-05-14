import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PATH = "./"
NFunc = 1
NRuns = 31
AllRes = np.zeros((NFunc,NRuns,1001))
for func in range(NFunc):
    AllRes[func] = np.loadtxt(PATH+f"L-SRTDE_GNBG_F{func+1}_D30_v2.txt")
 
fig = plt.figure(figsize=(15, 9), constrained_layout=True)
gs = fig.add_gridspec(6,4)
ax = []
for func in range(NFunc):
    ax.append(fig.add_subplot(gs[func]))
for func in range(NFunc):
    for r in range(NRuns):
        ax[func].plot(AllRes[func,r,:-1])
    ax[func].set_title(f"F{func+1}")
    ax[func].set_yscale("log")
fig.savefig(PATH+"L-SRTDE_GNBG_24_GRAPHS.pdf")
fig.savefig(PATH+"L-SRTDE_GNBG_24_GRAPHS.png")
plt.show()
