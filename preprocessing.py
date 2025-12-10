import numpy as np
import pyshtools as pysh
import os


def load_grail_sha_tab(path, lmax=None):
    """Load GRAIL/MESSENGER spherical harmonic coefficients from ascii tab raw data"""
    L_list, M_list, C_list, S_list = [], [], [], []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line_num == 1:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                l = int(parts[0])
                m = int(parts[1])
                C = float(parts[2])
                S = float(parts[3])
            except ValueError:
                continue

            if lmax is not None and l > lmax:
                continue

            L_list.append(l)
            M_list.append(m)
            C_list.append(C)
            S_list.append(S)

    if len(L_list) == 0:
        raise RuntimeError(f"No valid spherical harmonic rows in {path}")

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
    """Slice arrays to desired degree (more efficient than zeroing)"""
    return C[:l_trunc+1, :l_trunc+1].copy(), S[:l_trunc+1, :l_trunc+1].copy()


def make_grid_from_coeffs(C, S, grid_type="DH2", gm=4902.8005, r_ref=1737.4):
    """
    Convert spherical harmonic coefficients to gravity anomaly field

    Args:
        C, S: Coefficient matrices (dimensionless, fully normalized)
        grid_type: Grid type (DH2)
        gm: Gravitational parameter (km^3/s^2) - Moon: 4902.8005
        r_ref: Reference radius (km) - Moon: 1737.4
    """
    lmax = C.shape[0] - 1

    coeffs = pysh.SHCoeffs.from_array(
        np.array([C, S]),
        normalization='4pi',
        csphase=1,
        units='m'
    )

    C_scaled = C.copy()
    S_scaled = S.copy()

    for l in range(lmax + 1):
        factor = (l + 1)
        C_scaled[l, :] *= factor
        S_scaled[l, :] *= factor

    coeffs_gravity = pysh.SHCoeffs.from_array(
        np.array([C_scaled, S_scaled]),
        normalization='4pi',
        csphase=1
    )

    grid = coeffs_gravity.expand(grid=grid_type, extend=False)

    gravity_grid = grid.to_array() * (gm * 1e9) / ((r_ref * 1e3) ** 2) * 1e5

    return gravity_grid


def process_body(body_name, sha_file_path, max_degree, low_degrees, output_dir="data/processed"):
    """
    Process a celestial body's gravity data

    Args:
        body_name: Name of the body (e.g., 'moon', 'mercury')
        sha_file_path: Path to the spherical harmonic .tab file
        max_degree: Maximum degree to process (e.g., 200)
        low_degrees: List of low-degree resolutions (e.g., [25, 50])
        output_dir: Where to save processed files
    """
    os.makedirs(output_dir, exist_ok=True)

    C, S = load_grail_sha_tab(sha_file_path, lmax=max_degree)

    C_hi, S_hi = truncate_coeffs(C, S, max_degree)
    grid_hi = make_grid_from_coeffs(C_hi, S_hi)

    np.savez_compressed(
        f"{output_dir}/{body_name}_grav_L{max_degree}.npz",
        grid=grid_hi,
        lmax=max_degree
    )

    for L_low in low_degrees:
        C_low, S_low = truncate_coeffs(C, S, L_low)
        grid_low = make_grid_from_coeffs(C_low, S_low)

        np.savez_compressed(
            f"{output_dir}/{body_name}_grav_L{L_low}.npz",
            grid=grid_low,
            lmax=L_low
        )

    np.savez_compressed(
        f"{output_dir}/{body_name}_coeffs_raw.npz",
        C=C,
        S=S,
        lmax=max_degree
    )


if __name__ == "__main__":
    MAX_DEGREE = 600
    LOW_DEGREES = [25, 50, 100, 200]

    moon_file = "data/train/moon_large/jggrx_1800f_sha.tab"

    if os.path.exists(moon_file):
        process_body(
            body_name="moon",
            sha_file_path=moon_file,
            max_degree=MAX_DEGREE,
            low_degrees=LOW_DEGREES
        )

    mercury_file = "data/train/mercury/ggmes_100v08_sha.tab"

    if os.path.exists(mercury_file):
        process_body(
            body_name="mercury",
            sha_file_path=mercury_file,
            max_degree=100,
            low_degrees=LOW_DEGREES
        )
