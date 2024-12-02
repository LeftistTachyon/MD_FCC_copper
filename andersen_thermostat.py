import numpy as np

rng = np.random.default_rng()
k_B = 1.380_649e-23  # J/K


def init_velR(n: int, Tin: float):
    """
    Pick velocities from Maxwell-Boltzmann distribution
    for any temperature we want.
    Then we will calculate the kinetic energy and thus
    the temperature of these atoms and then we will
    rescale the velocities to the correct temperature

    Input:
    - n (integer): number of steps in trajectory
    - Tin (float): initial temperature
    Output:
    - vx, vy, vz (float): initial velocities
    - px, py, pz (float): initial momentums
    """
    k = 0

    # sample from M-B distribution via the double random method
    vx1 = rng.random(n)
    vx2 = rng.random(n)
    vx = np.sqrt(-2 * np.log(vx1)) * np.cos(2 * np.pi * vx2)
    vy1 = rng.random(n)
    vy2 = rng.random(n)
    vy = np.sqrt(-2 * np.log(vy1)) * np.cos(2 * np.pi * vy2)
    vz1 = rng.random(n)
    vz2 = rng.random(n)
    vz = np.sqrt(-2 * np.log(vz1)) * np.cos(2 * np.pi * vz2)

    # Find average momentum per atom
    px = vx.mean()
    py = vy.mean()
    pz = vz.mean()

    # Set net momentum to zero and calculate K
    vx -= px
    vy -= py
    vz -= pz

    k = np.sum(np.square(vx) + np.square(vy) + np.square(vz)) / 2

    # Kinetic energy of desired temperature (Tin)
    kin = 1.5 * n * Tin

    # Rescale velocities
    sc = np.sqrt(kin / k)
    vx *= sc
    vy *= sc
    vz *= sc

    return vx, vy, vz


def init_vel(n: int, Tin: float, m: float = 1.0):
    """
    Pick velocities from Maxwell-Boltzmann distribution
    for any temperature we want.
    Then we will calculate the kinetic energy and thus
    the temperature of these atoms and then we will
    rescale the velocities to the correct temperature

    Input:
    - n (integer): number of steps in trajectory
    - Tin (float): initial temperature
    - m (float): the mass of the particle
    Output:
    - vx, vy, vz (float): initial velocities
    - px, py, pz (float): initial momentums
    """
    k = 0

    # sample from M-B distribution via the double random method
    vx1 = rng.random(n)
    vx2 = rng.random(n)
    vx = np.sqrt(-2 * np.log(vx1)) * np.cos(2 * np.pi * vx2)
    vy1 = rng.random(n)
    vy2 = rng.random(n)
    vy = np.sqrt(-2 * np.log(vy1)) * np.cos(2 * np.pi * vy2)
    vz1 = rng.random(n)
    vz2 = rng.random(n)
    vz = np.sqrt(-2 * np.log(vz1)) * np.cos(2 * np.pi * vz2)

    # Find average momentum per atom
    px = vx.mean()
    py = vy.mean()
    pz = vz.mean()

    # Set net momentum to zero and calculate K
    vx -= px
    vy -= py
    vz -= pz

    k = m * np.sum(np.square(vx) + np.square(vy) + np.square(vz)) / 2

    # Kinetic energy of desired temperature (Tin)
    kin = 1.5 * n * Tin * k_B

    # Rescale velocities
    sc = np.sqrt(kin / k)
    vx *= sc
    vy *= sc
    vz *= sc

    return vx, vy, vz


def nudge(
    vx: np.ndarray[int, np.float64],
    vy: np.ndarray[int, np.float64],
    vz: np.ndarray[int, np.float64],
    coupling_constant: float,
    alpha: float,
):
    """
    ### Simulates the "collisions" needed to implement the Andersen thermostat

    Inputs:
    - vx: x velocities of each particle
    - vy: y velocities of each particle
    - vz: z velocities of each particle
    - coupling_constant: the coupling constant for the collisions
    - alpha: the scale of the normal distribution for the particle's velocities
    """
    vel = np.array([vx, vy, vz])
    l = vx.size
    vel = np.where(
        rng.random(l) < coupling_constant,
        rng.normal(loc=0, scale=alpha, size=(3, l)),
        vel,
    )

    return vel
