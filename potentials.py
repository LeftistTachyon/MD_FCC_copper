from numpy import exp


def forceLJ_r(dist: float) -> float:
    """
    Calculate the amount of force between two particles
    interacting via the Lennard-Jones potential
    depending on the given distance **in reduced units**

    Inputs:
    - dist (float): the distance between the particles

    Output:
    - (float): the force between the two particles
    """
    return 24 / dist**7 - 48 / dist**13


def potLJ_r(dist: float) -> float:
    """
    Calculate the potential difference between two
    particles as defined by the Lennard-Jones potential
    depending on the given distance **in reduced units**

    Inputs:
    - dist (float): the distance between the particles

    Output:
    - (float): the potential difference between the two particles
    """
    return 4 / dist**12 - 4 / dist**6


def forceLJ(dist: float, e: float = 6.567e-20, s: float = 2.334e-10) -> float:
    """
    Calculate the amount of force between two particles
    interacting via the Lennard-Jones potential
    depending on the given distance **in real units**

    `e` and `s` have default values for FCC Cu

    Inputs:
    - dist (float): the distance between the particles (in meters)
    - e (float): "epsilon" parameter in the LJ potential (in joules)
    - s (float): "sigma" parameter in the LJ potential (in angstroms)

    Output:
    - (float): the force between the two particles (in newtons)
    """
    temp = (s / dist) ** 6
    return e * (24 * temp - 48 * temp * temp) / dist


def potLJ(dist: float, e: float = 6.567e-20, s: float = 2.334e-10) -> float:
    """
    Calculate the potential difference between two
    particles as defined by the Lennard-Jones potential
    depending on the given distance **in real units**

    `e` and `s` have default values for FCC Cu

    Inputs:
    - dist (float): the distance between the particles (in meters)
    - e (float): "epsilon" parameter in the LJ potential (in joules)
    - s (float): "sigma" parameter in the LJ potential (in angstroms)

    Output:
    - (float): the potential difference between the two particles (in joules)
    """
    temp = (s / dist) ** 6
    return e * (4 * temp * temp - 4 * temp)


def forceMorse_r(dist: float) -> float:
    """
    Calculate the amount of force between two particles
    interacting via the Morse potential
    depending on the given distance **in reduced units**
    - r' = A * (r - r*) + 1

    Inputs:
    - dist (float): the distance between the particles

    Output:
    - (float): the force between the two particles
    """
    return 2 * (exp(1 - dist) - exp(2 - 2 * dist))


def potMorse_r(dist: float) -> float:
    """
    Calculate the potential difference between two
    particles as defined by the Morse potential
    depending on the given distance **in reduced units**
    - r' = A * (r - r*) + 1

    Inputs:
    - dist (float): the distance between the particles

    Output:
    - (float): the potential difference between the two particles
    """
    return exp(2 - 2 * dist) - 2 * exp(1 - dist)


def forceMorse(
    dist: float, e: float = 5.292e-20, A: float = 1.329e10, rmin: float = 2.885e-10
) -> float:
    """
    Calculate the amount of force between two particles
    interacting via the Morse potential
    depending on the given distance **in real units**

    `e`, `A`, and `rmin` have default values for FCC Cu

    Inputs:
    - dist (float): the distance between the particles (in meters)
    - e (float): the "epsilon" parameter for the Morse potential (in joules)
    - A (float): the "A" or "alpha" parameter for the Morse potential (in inverse meters)
    - rmin (float): the "rmin" parameter for the Morse potential (in meters)

    Output:
    - (float): the force between the two particles (in newtons)
    """
    temp = A * (dist - rmin)
    return e * A * 2 * (exp(-temp) - exp(-2 * temp))


def potMorse(
    dist: float, e: float = 5.292e-20, A: float = 1.329e10, rmin: float = 2.885e-10
) -> float:
    """
    Calculate the potential difference between two
    particles as defined by the Morse potential
    depending on the given distance **in real units**

    Inputs:
    - dist (float): the distance between the particles (in meters)
    - e (float): the "epsilon" parameter for the Morse potential (in joules)
    - A (float): the "A" or "alpha" parameter for the Morse potential (in inverse meters)
    - rmin (float): the "rmin" parameter for the Morse potential (in meters)

    Output:
    - (float): the potential difference between the two particles (in joules)
    """
    temp = A * (dist - rmin)
    return e * (exp(-2 * temp) - 2 * exp(-temp))


# the reduced unit constant for copper
_cu_const = 1.329e10 * 2.885e-10


def forceMorse_rCu(dist: float, Ar: float = _cu_const) -> float:
    """
    Calculate the amount of force between two particles
    interacting via the Morse potential
    depending on the given distance **in semi-reduced units**

    `Ar` has the default value for FCC Cu

    Inputs:
    - dist (float): the distance between the particles (reduced units)
    - Ar (float): the reduced "A" or "alpha" parameter for the Morse potential (dimensionless)

    Output:
    - (float): the force between the two particles (in newtons)
    """
    temp = Ar * (1 - dist)
    return Ar * 2 * (exp(temp) - exp(2 * temp))


def potMorse_rCu(dist: float, Ar: float = _cu_const) -> float:
    """
    Calculate the potential difference between two
    particles as defined by the Morse potential
    depending on the given distance **in semi-reduced units**

    Inputs:
    - dist (float): the distance between the particles (reduced units)
    - Ar (float): the reduced "A" or "alpha" parameter for the Morse potential (dimensionless)

    Output:
    - (float): the potential difference between the two particles (in joules)
    """
    temp = Ar * (1 - dist)
    return exp(2 * temp) - 2 * exp(temp)
