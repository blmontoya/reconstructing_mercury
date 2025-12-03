import numpy as np
import pyshtools as pysh
import os

# load coeffs from tab data
def load_grail_sha_tab(path, lmax=None):

    #load GRAIL slpherical harmonic coefficients from ascii tab raw data
    L_list, M_list, C_list, S_list = [], [], [], []

    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # skip the 1-row SHADR header table
            if line_num == 1:
                continue

            #PDS3 ASCII format
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            try:
                l = int(parts[0])
                m = int(parts[1])
                C = float(parts[2])
                S = float(parts[3])
            except ValueError:
                print("this shouldn't've happened ):")
                continue

            L_list.append(l)
            M_list.append(m)
            C_list.append(C)
            S_list.append(S)

    if len(L_list) == 0:
        raise RuntimeError(f"No valid spherical harmonic rows in {path}")

    # infer lmax if not given
    if lmax is None:
        lmax = max(L_list)

    Cmat = np.zeros((lmax + 1, lmax + 1))
    Smat = np.zeros((lmax + 1, lmax + 1))

    for l, m, C, S in zip(L_list, M_list, C_list, S_list):
        if l <= lmax and m <= l:
            Cmat[l, m] = C
            Smat[l, m] = S

    return Cmat, Smat

def truncate_coeffs(C, S, l_trunc):
    C_low = C.copy()
    S_low = S.copy()
    C_low[l_trunc+1:, :] = 0
    S_low[l_trunc+1:, :] = 0
    return C_low, S_low

#convert coefficients to grid
def make_grid_from_coeffs(C, S, grid_type="DH2"):

    #convert spherical harmonic coefficient matrices into a numpy grid for input to CNN
    #trying to input pure spherical harmonic data to CNN would be hell

    coeffs = pysh.SHCoeffs.from_array(np.array([C, S]))
    grid = coeffs.expand(grid=grid_type)#this method is extremely annoying, dont touch ideally
    return grid.to_array()

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("Loading high-degree lunar model...")
    C, S = load_grail_sha_tab("data/train/moon_large/jggrx_1800f_sha.tab")

    #high degree
    L_hi = 600
    print(f"generating grid (L={L_hi})...")
    C_hi, S_hi = truncate_coeffs(C, S, L_hi)
    grid_hi = make_grid_from_coeffs(C_hi, S_hi)

    np.save("data/processed/moon_grav_L600.npy", grid_hi)

    #form low degree inputs as well
    for L_low in [25, 50, 100]:#extremely arbitrary we'll see whats of use later while training/what fits
        print(f"Generating low-degree grid (L={L_low})...")
        C_low, S_low = truncate_coeffs(C, S, L_low)
        grid_low = make_grid_from_coeffs(C_low, S_low)
        np.save(f"data/processed/moon_grav_L{L_low}.npy", grid_low)

    print("finished, saved all grids")
