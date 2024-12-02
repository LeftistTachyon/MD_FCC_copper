import matplotlib.pyplot as plt
import numpy as np
from potentials import potLJ_r, potMorse_rCu

# graphing parameters
plt.rcParams.update({"font.size": 22})

# prepare data
x_lj = np.arange(0.95, 2, 0.025)
x_m = np.arange(0.75, 2, 0.025)
U_lj = potLJ_r(x_lj)
U_m = potMorse_rCu(x_m)

# plot
print("plotting & saving energies...", end=" ")
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x_lj, U_lj, color="b", linewidth=1, label="Lennard-Jones potential")
ax2.plot(x_m, U_m, color="r", linewidth=1, label="Morse potential")

ax1.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax2.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax1.set_xlabel("separation distance (reduced units)")
ax1.set_ylabel("LJ potential energy\n(reduced units)")
ax2.set_xlabel("separation distance (reduced units)")
ax2.set_ylabel("Morse potential energy\n(reduced units)")

fig.tight_layout()
plt.show()
fig.savefig("FinalProject/comparison.svg")
plt.close(fig)
print("saved!")
