import sys
from typing import Callable

import numpy as np
from andersen_thermostat import init_velR, nudge
from gen_latt import load_disturbedfcclattice, unpack_coordinates
from potentials import forceMorse_rCu, potMorse_rCu, forceLJ_r, potLJ_r
from rad_dist import rad_dist, trans_ordparam
from tqdm.auto import tqdm


def MDv_reduced(
    nc: int,
    density: float,
    tin: float,
    nsteps: int,
    dt: float,
    disturbance_strength: int = 8,
    coupling_const: float = 0.025,
    force_function: Callable[
        [int, np.ndarray[np.float64]], np.ndarray[int, np.float64]
    ] = forceMorse_rCu,
    pot_function: Callable[
        [int, np.ndarray[np.float64]], np.ndarray[int, np.float64]
    ] = potMorse_rCu,
    checkpoint_freq: int = 250,
):
    """
    ## Performs a MD simulation with an Andersen thermostat
    With the given simulation cell and environment parameters, the system will attempt to find equilibrium.

    Simulation steps:
    - Initialize simulation cell /w positions and velocities
    - Repeatedly calculate energies, then next position & velocities (via velocity Verlet)

    Inputs:
    - nc (integer): number of unit cells
    - density (float): volume density in reduced units
    - tin (float): intput temperature in reduced units
    - nsteps (integer): total number of time steps
    - dt (float): time step in reduced units
    - disturbance_strength (int): how much the lattice should be disturbed from the standard lattice position (larger = less disturbance)
    - coupling_const (float): the coupling constant for the Andersen thermostat
    - force_function (float -> float, vectorized): the function to determine the amount of force between two particles (distance-dependant)
    - pot_function (float -> float, vectorized): the function to determine the potential difference between two particles (distance-dependant)
    - checkpoint_freq (int): how many sim steps should pass before displaying a printout of all relevant values

    Outputs (as tuple):
    1. un: a time-series list of the potential energies of the system
    2. kn: a time-series list of the kinetic energies of the system
    3. en: a time-series list of the total energies of the system
    4. tn: a time-series list of the temperatures of the system
    5. pn: a time-series list of the pressures of the system
    6. ton: a time-series list of the traslational order parameters of the system
    7. ton: a time-series list of the traslational order parameters (3D ver.) of the system
    7. gr: the radial distribution function of the final state of the system
    """

    # Define helper function for calculating forces and other energies
    def forces(
        a: float,
        x: np.ndarray[int, np.float64],
        y: np.ndarray[int, np.float64],
        z: np.ndarray[int, np.float64],
    ):
        """
        Simple lattice sum for force with cutoffs and
        minimum image convention

        We calculate force (fx, fy, fz), energy (u), and
        part of the pressure (w).

        """
        # calculate displacements via minimum image convention
        dx = np.meshgrid(x, x, indexing="ij")
        dx = dx[1] - dx[0]
        dx -= np.round(dx)
        dy = np.meshgrid(y, y, indexing="ij")
        dy = dy[1] - dy[0]
        dy -= np.round(dy)
        dz = np.meshgrid(z, z, indexing="ij")
        dz = dz[1] - dz[0]
        dz -= np.round(dz)

        # calculate distances (& normalize as needed)
        dist = a * np.sqrt(dx**2 + dy**2 + dz**2)
        dist_n = np.where(dist == 0, np.inf, dist)  # prevent 0 from blowing things up

        dphi = force_function(dist_n)
        unit_f = dphi * a / dist_n
        ffx = unit_f * dx
        ffy = unit_f * dy
        ffz = unit_f * dz
        phi = pot_function(dist_n)

        # print(dphi * dist)
        # print(48 / dist_n**12 - 24 / dist_n**6)
        # print(ffx * dx + ffy * dy + ffz * dz)

        # divide all quantities by 2 to prevent overcounting
        u = phi.sum() / 2
        w = np.sum(-dphi * dist) / 2
        fx = ffx.sum(axis=1) - ffx.sum(axis=0)
        fy = ffy.sum(axis=1) - ffy.sum(axis=0)
        fz = ffz.sum(axis=1) - ffz.sum(axis=0)

        sys.exit()

        return u, w, fx, fy, fz

    # Initialize positions and velocities
    n, x, y, z = unpack_coordinates(
        load_disturbedfcclattice(nc, disturbance_strength=disturbance_strength)
    )
    vx, vy, vz = init_velR(n, tin)

    # Calculate some useful quantities
    vol = n / density
    a = np.cbrt(vol)
    # print(f"chosen a: {a:.5f}")

    # Calculate initial energy and forces
    u, w, fx, fy, fz = forces(a, x, y, z)

    # Time series arrays
    un = np.zeros(nsteps)
    kn = np.zeros(nsteps)
    en = np.zeros(nsteps)
    tn = np.zeros(nsteps)
    pn = np.zeros(nsteps)
    ton = np.zeros(nsteps)

    # some constants
    c1 = dt / a
    c2 = dt**2 / (2 * a)
    c3 = dt / 2

    # Start the time steps
    for j in tqdm(range(nsteps), desc="Simulating", colour="blue"):
        # Calculate ensemble values
        k = np.sum(vx**2 + vy**2 + vz**2) / 2
        temp = k / (1.5 * n)

        # Create time series of values
        e = k + u
        un[j] = u / n
        kn[j] = k / n
        en[j] = e / n
        tn[j] = temp
        pn[j] = density * temp + w / (3 * vol)
        ton[j] = trans_ordparam(nc, x, y, z)

        # Find x(t + dt)
        x += vx * c1 + fx * c2
        y += vy * c1 + fy * c2
        z += vz * c1 + fz * c2

        # Calculate force and energy at new positions for next cycle
        u, w, fxnew, fynew, fznew = forces(a, x, y, z)

        # Find v(t + dt)
        vx += (fx + fxnew) * c3
        vy += (fy + fynew) * c3
        vz += (fz + fznew) * c3

        # Andersen thermostat "collisions"
        vx, vy, vz = nudge(
            vx,
            vy,
            vz,
            coupling_constant=coupling_const,
            alpha=np.sqrt(tin),
        )

        # Find a(t + dt)
        fx, fy, fz = fxnew, fynew, fznew

        # log results to make it not look like everything is hung
        if checkpoint_freq and j % checkpoint_freq == 0:
            print(f"\nresults on step {j}: ({j*100/nsteps:.3f}% complete)")
            print(f"  potential energy: {un[j]:+.3f}")
            print(f"  kinetic energy  : {kn[j]:+.3f}")
            print(f"  total energy    : {en[j]:+.3f}")
            print(f"  temperature     : {tn[j]:+.3f}")
            print(f"  pressure        : {pn[j]:+.3f}")
            print(f"  trans ord param : {ton[j]:+.3f}")

    # final: radial distribution function
    gr = rad_dist(a, x, y, z)

    # import matplotlib.pyplot as plt

    # print("plotting & saving config...", end=" ")
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(x, y, z, color="k", marker="o")

    # fig.tight_layout()
    # plt.show()
    # fig.savefig(f"FinalProject/last_config.svg")
    # plt.close(fig)
    # print("saved!")

    return un, kn, en, tn, pn, ton, gr


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # graphing parameters
    plt.rcParams.update({"font.size": 22})

    # simulation parameters
    total_time = 50
    dt = 0.0025

    # derived parameters
    t = np.arange(0, total_time, dt)

    # run the simulation
    un, kn, en, tn, pn, ton, gr = MDv_reduced(
        force_function=forceLJ_r,
        pot_function=potLJ_r,
        nc=1,
        density=3.8,
        tin=2.4,
        nsteps=int(total_time / dt),
        dt=dt,
        checkpoint_freq=0,
    )

    print("plotting & saving energies...", end=" ")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(t, un, "b-", lw=2, label="Potential energy")
    ax2.plot(t, kn, "r-", lw=2, label="Kinetic energy")
    ax3.plot(t, en, "k-", lw=2, label="Total energy")

    ax1.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax1.set_xlabel("Time (reduced units)")
    ax1.set_xlim([t[0], t[-1]])
    ax1.set_ylabel("Potential energy\n(reduced units)")
    ax2.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax2.set_xlabel("Time (reduced units)")
    ax2.set_xlim([t[0], t[-1]])
    ax2.set_ylabel("Kinetic energy\n(reduced units)")
    ax3.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax3.set_xlabel("Time (reduced units)")
    ax3.set_xlim([t[0], t[-1]])
    ax3.set_ylabel("Total energy\n(reduced units)")

    fig.tight_layout()
    plt.show()
    fig.savefig("FinalProject/sample_verlet_energies.svg")
    plt.close(fig)
    print("saved!")

    print("plotting & saving temps...", end=" ")
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, tn, color="k", lw=2, label="Temperature")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_xlabel("Time (reduced units)")
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylabel("Temperature (reduced units)")

    fig.tight_layout()
    plt.show()
    fig.savefig("FinalProject/sample_verlet_temps.svg")
    plt.close(fig)
    print("saved!")

    print("plotting & saving pressures...", end=" ")
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, pn, color="k", lw=2, label="Pressure")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_xlabel("Time (reduced units)")
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylabel("Pressure (reduced units)")

    fig.tight_layout()
    plt.show()
    fig.savefig("FinalProject/sample_verlet_pressures.svg")
    plt.close(fig)
    print("saved!")

    print("plotting & saving translational order parameters...", end=" ")
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, ton, color="r", lw=2, label="std")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_xlabel("Time (reduced units)")
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylabel("Translational order parameter")

    fig.tight_layout()
    plt.show()
    fig.savefig("FinalProject/sample_verlet_transordparams.svg")
    plt.close(fig)
    print("saved!")

    print("plotting & saving radial distribution...", end=" ")
    ng, binedges = gr
    fig, ax = plt.subplots(1, 1)
    ax.bar(
        binedges[:-1],
        ng,
        width=binedges[1] - binedges[0],
        label="Radial distribution",
    )
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.set_xlabel("Distance (reduced units)")
    ax.set_xlim([binedges[0], binedges[-1]])
    ax.set_ylabel("Radial distribution")

    # Horizontal line at g(r) = 1
    ax.axhline(y=1, color="r", linestyle="--")

    fig.tight_layout()
    plt.show()
    fig.savefig("FinalProject/sample_verlet_radialdist.svg")
    plt.close(fig)
    print("saved!")

    # print("\nreports:")
    # energy_deviation = (en.max() - en.min()) / en.mean()
    # print(f"  energy deviation (ratio): {energy_deviation:.8f}")
    # energy_std = en.std() / en.mean()
    # print(f"  energy stdev (ratio): {energy_std:.8f}")
    # k_peak1 = kn.max() - kn[0]
    # print(f"  k peak: {k_peak1}")
    # u_dip1 = un[0] - un.min()
    # print(f"  u dip : {u_dip1}")
    # print(f"  ratio : {k_peak1 / u_dip1}")

    print("saving to csv file...", end=" ")
    csv_data = np.array([t, un, kn, en, tn, pn, ton]).T
    np.savetxt(
        "FinalProject/sample_data.csv",
        csv_data,
        delimiter=",",
        header="Time,Potential energy,Kinetic energy,Total energy,Temperature,Pressure,Translational order parameter",
    )
    print("done!")


def MDv_reduced_minimal(
    nc: int,
    density: float,
    tin: float,
    nsteps: int,
    dt: float,
    disturbance_strength: int = 8,
    coupling_const: float = 0.025,
    force_function: Callable[
        [int, np.ndarray[np.float64]], np.ndarray[int, np.float64]
    ] = forceMorse_rCu,
    save_cutoff: float = 0.45,
):
    """
    ## Performs a MD simulation with an Andersen thermostat
    With the given simulation cell and environment parameters, the system will attempt to find equilibrium.

    NOTE: the simulation will give only bare-bones information about the system. For all information, see MDv_reduced.

    Simulation steps:
    - Initialize simulation cell /w positions and velocities
    - Repeatedly calculate energies, then next position & velocities (via velocity Verlet)

    Inputs:
    - nc (integer): number of unit cells
    - density (float): volume density in reduced units
    - tin (float): intput temperature in reduced units
    - nsteps (integer): total number of time steps
    - dt (float): time step in reduced units
    - disturbance_strength (int): how much the lattice should be disturbed from the standard lattice position (larger = less disturbance)
    - coupling_const (float): the coupling constant for the Andersen thermostat
    - force_function (float -> float, vectorized): the function to determine the amount of force between two particles (distance-dependant)
    - save_cutoff (float): the ratio of first runs to discard before beginning to average

    Outputs (as tuple):
    1. tmean: the mean temperature of the system after equilibration
    2. pmean: the mean pressure of the system after equilibration
    3. tomean: the mean traslational order parameter of the system after equilibration
    4. gr: the radial distribution function of the final state of the system
    """

    # Define helper function for calculating forces and other energies
    def forces(
        a: float,
        x: np.ndarray[int, np.float64],
        y: np.ndarray[int, np.float64],
        z: np.ndarray[int, np.float64],
    ):
        """
        Simple lattice sum for force with cutoffs and
        minimum image convention

        We calculate force (fx, fy, fz), energy (u), and
        part of the pressure (w).

        """
        # calculate displacements via minimum image convention
        dx = np.meshgrid(x, x, indexing="ij")
        dx = dx[1] - dx[0]
        dx -= np.round(dx)
        dy = np.meshgrid(y, y, indexing="ij")
        dy = dy[1] - dy[0]
        dy -= np.round(dy)
        dz = np.meshgrid(z, z, indexing="ij")
        dz = dz[1] - dz[0]
        dz -= np.round(dz)

        # calculate distances (& normalize as needed)
        dist = a * np.sqrt(dx**2 + dy**2 + dz**2)
        dist_n = np.where(dist == 0, np.inf, dist)  # prevent 0 from blowing things up

        dphi = force_function(dist_n)
        unit_f = dphi * a / dist_n
        ffx = unit_f * dx
        ffy = unit_f * dy
        ffz = unit_f * dz

        # divide all quantities by 2 to prevent overcounting
        w = np.sum(-dphi * dist) / 2
        fx = ffx.sum(axis=1) - ffx.sum(axis=0)
        fy = ffy.sum(axis=1) - ffy.sum(axis=0)
        fz = ffz.sum(axis=1) - ffz.sum(axis=0)

        return w, fx, fy, fz

    # Initialize positions and velocities
    n, x, y, z = unpack_coordinates(
        load_disturbedfcclattice(nc, disturbance_strength=disturbance_strength)
    )
    vx, vy, vz = init_velR(n, tin)

    # Calculate some useful quantities
    vol = n / density
    a = np.cbrt(vol)

    # Calculate initial energy and forces
    w, fx, fy, fz = forces(a, x, y, z)

    # Time series arrays
    to_discard = int(save_cutoff * nsteps)
    to_save = nsteps - to_discard
    tn = np.zeros(to_save)
    pn = np.zeros(to_save)
    ton = np.zeros(to_save)

    # some constants
    c1 = dt / a
    c2 = dt**2 / (2 * a)
    c3 = dt / 2

    # Start the time steps
    for j in tqdm(range(nsteps), desc="Simulating", colour="blue", leave=False):
        if j >= to_discard:
            # Calculate ensemble values
            k = np.sum(vx**2 + vy**2 + vz**2) / 2
            temp = k / (1.5 * n)

            # Create time series of values
            idx = j - to_discard
            tn[idx] = temp
            pn[idx] = density * temp + w / (3 * vol)
            ton[idx] = trans_ordparam(nc, x, y, z)

        # Find x(t + dt)
        x += vx * c1 + fx * c2
        y += vy * c1 + fy * c2
        z += vz * c1 + fz * c2

        # Calculate force and energy at new positions for next cycle
        w, fxnew, fynew, fznew = forces(a, x, y, z)

        # Find v(t + dt)
        vx += (fx + fxnew) * c3
        vy += (fy + fynew) * c3
        vz += (fz + fznew) * c3

        # Andersen thermostat "collisions"
        vx, vy, vz = nudge(
            vx,
            vy,
            vz,
            coupling_constant=coupling_const,
            alpha=np.sqrt(tin),
        )

        # Find a(t + dt)
        fx, fy, fz = fxnew, fynew, fznew

    # final: radial distribution function
    gr = rad_dist(a, x, y, z)

    return tn.mean(), pn.mean(), np.abs(ton).mean(), gr
