import numpy as np


def rad_dist(
    a: float,
    x: np.ndarray[int, np.float64],
    y: np.ndarray[int, np.float64],
    z: np.ndarray[int, np.float64],
    num_bins: int = 250,
):
    """
    Calculate radial distribution function.

    Input:
        a (float): simulation cell dimension
        x, y, z (array of floats): atomic positions
        num_bins (int): the number of bins to use
    Output:
        ng (array): frequencies corresponding to g(r)
        bp (array): binning used in g(r)
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

    # histogram distances
    hist, bin_edges = np.histogram(dist, bins=np.linspace(0, a / 2, num_bins + 1))

    # normalize
    factor = a**3 / (4 * np.pi * len(x) ** 2 * (bin_edges[1] - bin_edges[0]))
    ng = factor * hist / np.concatenate(([np.inf], bin_edges[1:-1])) ** 2

    # return values
    return ng, bin_edges


def rad_dist2(
    a: float,
    x: np.ndarray[int, np.float64],
    y: np.ndarray[int, np.float64],
    z: np.ndarray[int, np.float64],
    bin_width: float = 0.01,
):
    """
    Calculate radial distribution function v2.

    Input:
        a (float): simulation cell dimension
        x, y, z (array of floats): atomic positions
        num_bins (int): the number of bins to use
    Output:
        g (array): frequencies corresponding to g(r)
        r (array): binning used in g(r)
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

    # bin distances
    bins = np.arange(bin_width, a / 2 + bin_width * 2, bin_width)
    hist, _ = np.histogram(dist, bins=bins)

    rho = len(x) / (3 * a) ** 3
    r = (np.arange(1, len(hist) + 1) - 0.5) * bin_width
    shell_volumes = 4 * np.pi * r**2 * bin_width
    g = hist / (2 * rho * len(x) * shell_volumes)
    return r, g


def trans_ordparam(
    nc: int,
    x: np.ndarray[int, np.float64],
    y: np.ndarray[int, np.float64],
    z: np.ndarray[int, np.float64],
    k: np.ndarray[int, np.float64] = 2 * np.pi * np.array([-1, 1, -1]),
):
    """
    Calculates the translational order parameter for the given particles.
    Assumes the typical order is an FCC lattice (for defining k).

    Inputs:
    - nc (int): the number of cells in the simulation cell (scale parameter)
    - x (array of floats): the x coordinates of the particles
    - y (array of floats): the y coordinates of the particles
    - z (array of floats): the z coordinates of the particles
    - k (3D vector): the reciprocal lattice constant (default: FCC)
    """
    # pack the coordinates
    coords = np.array((x - np.round(x), y - np.round(y), z - np.round(z))).T
    # calculate parameter
    return np.cos(np.dot(coords, k) * nc).mean()


if __name__ == "__main__":
    from gen_latt import load_disturbedfcclattice, unpack_coordinates, load_fcclattice
    import matplotlib.pyplot as plt

    coords = load_fcclattice(4)
    n, x, y, z = unpack_coordinates(coords)
    print(f"\nTesting {n} FCC lattice points")

    print("radial distribution function:")
    r, g = rad_dist2(1, x, y, z)
    # Plot the radial distribution function
    plt.plot(r, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")

    # Horizontal line at g(r) = 1
    plt.axhline(y=1, color="r", linestyle="--")
    plt.text(0, 1.1, "Ideal Gas", color="red")

    plt.show()

    print("translational order parameter:", trans_ordparam(4, x, y, z))

    coords = load_disturbedfcclattice(4)
    n, x, y, z = unpack_coordinates(coords)
    print(f"\nTesting {n} disturbed FCC lattice points")

    print("radial distribution function:")
    r, g = rad_dist2(1, x, y, z)
    # Plot the radial distribution function
    plt.plot(r, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")

    # Horizontal line at g(r) = 1
    plt.axhline(y=1, color="r", linestyle="--")
    plt.text(0, 1.1, "Ideal Gas", color="red")

    plt.show()

    print("translational order parameter:", trans_ordparam(4, x, y, z))
