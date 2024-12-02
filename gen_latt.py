import numpy as np

rng = np.random.default_rng()
log = True


## Code for generating an FCC supercell
def fccmke(nc: int):
    """### Generate fractional coordinates of atoms in FCC supercell
    Input:
    - nc (integer): supercell of nc x nc x nc dimensions

    Output:
    - s (array): fractional atomic coordinates of supercell
    """
    r = np.array([[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
    i1 = 0
    s = np.zeros((len(r) * nc**3, 3))
    # in fractional coordinates
    for k in range(1, nc + 1):
        for l in range(1, nc + 1):
            for m in range(1, nc + 1):
                for i in range(len(r)):
                    s[i1, 0] = (r[i, 0] + k - 1) / nc
                    s[i1, 1] = (r[i, 1] + l - 1) / nc
                    s[i1, 2] = (r[i, 2] + m - 1) / nc
                    i1 += 1
    return s


def load_fcclattice(lattice_size: int):
    """
    ### Loads a pregenerated FCC lattice from the hard disk
    Inputs:
    - lattice_size (integer): the lattice size parameter to pass to the lattice generation function
    """
    try:
        # ...from file
        if log:
            print("reading lattice from file...", end=" ")
        lattice = np.load(f"FinalProject/lattice_fcc{lattice_size}.npy")
        if log:
            print("finished!")
    except:
        # ...or manually
        if log:
            print("failed, generating lattice...", end=" ")
        lattice = fccmke(lattice_size)
        if log:
            print("finished!")
        np.save(
            f"FinalProject/lattice_fcc{lattice_size}.npy", lattice
        )  # save for later
        if log:
            print("wrote lattice to file")

    return lattice


def load_disturbedfcclattice(lattice_size: int, disturbance_strength: float = 8):
    """
    ### Loads a pregenerated FCC lattice from the hard disk with additional disturbance
    Inputs:
    - lattice_size (integer): the lattice size parameter to pass to the lattice generation function
    - disturbance_strength (integer): the amount to disturb the lattice (higher = less disturbance)
    """

    lattice = load_fcclattice(lattice_size)
    return lattice + rng.random(lattice.shape) / (lattice_size * disturbance_strength)


def unpack_coordinates(lattice: np.ndarray[tuple[int, int], np.float64]):
    """
    ### Unpacks a 2D array of coordinates into three arrays of coordinates along with its size
    Inputs:
    - lattice (2d array of floats): the coordinates to unpack
    Outputs:
    - n (int): the number of coordinates found in the given array
    - x (int): the x coordinates
    - y (int): the y coordinates
    - z (int): the z coordinates
    """
    x, y, z = lattice.T
    return len(x), x, y, z


# running as script
if __name__ == "__main__":
    print(load_disturbedfcclattice(1))
