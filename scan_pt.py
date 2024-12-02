import concurrent.futures
import os
import shutil
from datetime import datetime
import time

import gen_latt
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from verlet import MDv_reduced_minimal

# global params
gen_latt.log = False

# simulation parameters
total_time = 50
dt = 0.0025
densities = np.arange(3.1, 3.51, 0.1)
temperatures = np.arange(3.125, 3.51, 0.125)
nc = 4

# derived parameters
t = np.arange(0, total_time, dt)

# sim result storage
t_l = temperatures.size
d_l = densities.size
mean_t = np.zeros((d_l, t_l))
mean_p = np.zeros((d_l, t_l))
mean_to = np.zeros((d_l, t_l))

# preparing output directory
global_output = f"FinalProject/scan{nc}"
if os.path.isdir(global_output):
    print("previous data detected; zipping & clearing...", end=" ")
    datestring = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.make_archive(
        f"FinalProject/savestate{nc}_{datestring}", "zip", global_output
    )
    shutil.rmtree(global_output)
    print("done")


# helper function: run 1 sim
def one_sim(i: int, j: int):
    # just to make sure the loading bars are drawn correctly
    time.sleep(0.01)

    # create folder
    output_folder = f"{global_output}/{densities[i]*100:03.0f}"
    os.makedirs(output_folder, exist_ok=True)

    # run sim
    tmean, pmean, to, gr = MDv_reduced_minimal(
        nc=nc,
        density=densities[i],
        tin=temperatures[j],
        nsteps=int(total_time / dt),
        dt=dt,
    )

    # logging data
    mean_t[i][j] = tmean
    mean_p[i][j] = pmean
    mean_to[i][j] = to

    # graph
    # print("plotting & saving radial distribution...", end=" ")
    return f"{output_folder}/raddist_{j}_{temperatures[j]*100:03.0f}.svg", gr
    # print("saved!")


# simulation loop
sim_space = np.meshgrid(range(d_l), range(t_l), indexing="ij")
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = list(
        tqdm(
            executor.map(one_sim, sim_space[0].ravel(), sim_space[1].ravel()),
            total=d_l * t_l,
            desc="Scanning",
            colour="yellow",
        )
    )


for file_name, gr in tqdm(results, desc="Plotting", colour="green"):
    ng, binedges = gr
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.bar(
        binedges[:-1],
        ng,
        width=binedges[1] - binedges[0],
        label="Radial distribution",
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set(
        xlabel="Distance (reduced units)",
        xlim=[binedges[0], binedges[-1]],
        ylabel="Radial distribution",
    )

    # Horizontal line at g(r) = 1
    ax.axhline(y=1, color="r", linestyle="--")

    fig.tight_layout()
    # plt.show()
    fig.savefig(file_name)
    plt.close(fig)

# tile x & y
tiled_densities = np.tile(densities, (t_l, 1)).T
tiled_temps = np.tile(temperatures, (d_l, 1))

print("plotting & saving temperatures...", end=" ")
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    tiled_densities, tiled_temps, mean_t, color="k", edgecolor="k", lw=0.5, alpha=0.3
)
ax.plot_surface(
    tiled_densities,
    tiled_temps,
    tiled_temps,
    color="r",
    edgecolor="r",
    lw=0.5,
    alpha=0.2,
)
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(
    xlim=[densities[0], densities[-1]],
    xlabel="Density input (reduced units)",
    ylim=[temperatures[0], temperatures[-1]],
    ylabel="Temperature input (reduced units)",
    zlabel="Mean temperature (reduced units)",
)

fig.tight_layout()
plt.show()
fig.savefig(f"{global_output}/temperatures.svg")
plt.close(fig)
print("saved!")


print("plotting & saving pressures...", end=" ")
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    tiled_densities, tiled_temps, mean_p, color="k", edgecolor="k", lw=0.5, alpha=0.3
)
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(
    xlim=[densities[0], densities[-1]],
    xlabel="Density input (reduced units)",
    ylim=[temperatures[0], temperatures[-1]],
    ylabel="Temperature input (reduced units)",
    zlabel="Mean pressure (reduced units)",
)

fig.tight_layout()
plt.show()
fig.savefig(f"{global_output}/pressures.svg")
plt.close(fig)
print("saved!")


print("plotting & saving order parameters...", end=" ")
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    tiled_densities, tiled_temps, mean_to, color="k", edgecolor="k", lw=0.5, alpha=0.3
)
ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
ax.set(
    xlim=[densities[0], densities[-1]],
    xlabel="Density input (reduced units)",
    ylim=[temperatures[0], temperatures[-1]],
    ylabel="Temperature input (reduced units)",
    zlabel="Mean translational order parameter",
)

fig.tight_layout()
plt.show()
fig.savefig(f"{global_output}/transorder.svg")
plt.close(fig)
print("saved!")

print("saving to csv file...", end=" ")
csv_data = np.array(
    [
        tiled_densities.ravel(),
        tiled_temps.ravel(),
        mean_t.ravel(),
        mean_p.ravel(),
        mean_to.ravel(),
    ]
).T
np.savetxt(
    f"{global_output}/data.csv",
    csv_data,
    delimiter=",",
    header="Input density,Input temperature,Mean temperature,Mean pressure,Mean translational order parameter",
)
print("done!")
