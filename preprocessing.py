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
            # Skip the 1-row SHADR header table
            if line_num == 1:
                continue
            # PDS3 ASCII format
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                l = int(parts[0])
                m = int(parts[1])
                C = float(parts[2])
                S = float(parts[3])
            except ValueError:
                print(f"Skipping invalid line {line_num}")
                continue
            
            # Early exit if we've loaded enough data
            if lmax is not None and l > lmax:
                continue
            
            L_list.append(l)
            M_list.append(m)
            C_list.append(C)
            S_list.append(S)
    
    if len(L_list) == 0:
        raise RuntimeError(f"No valid spherical harmonic rows in {path}")
    
    # Infer lmax if not given
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


def make_grid_from_coeffs(C, S, grid_type="DH2"):
    """Convert spherical harmonic coefficient matrices into a numpy grid"""
    coeffs = pysh.SHCoeffs.from_array(np.array([C, S]))
    grid = coeffs.expand(grid=grid_type)
    return grid.to_array()


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
    print("\n" + "="*80)
    print(f"PROCESSING {body_name.upper()} DATA")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load coefficients up to max degree needed
    print(f"\nLoading {body_name} model up to L={max_degree}...")
    C, S = load_grail_sha_tab(sha_file_path, lmax=max_degree)
    print(f"  Loaded coefficient matrices: {C.shape}")
    
    # Generate high-resolution grid
    print(f"\nGenerating high-res grid (L={max_degree})...")
    C_hi, S_hi = truncate_coeffs(C, S, max_degree)
    grid_hi = make_grid_from_coeffs(C_hi, S_hi)
    
    np.savez_compressed(
        f"{output_dir}/{body_name}_grav_L{max_degree}.npz",
        grid=grid_hi,
        lmax=max_degree
    )
    print(f"  Saved: {grid_hi.shape}, {grid_hi.nbytes / 1e6:.2f} MB")
    
    # Generate low-resolution grids
    for L_low in low_degrees:
        print(f"\nGenerating low-degree grid (L={L_low})...")
        C_low, S_low = truncate_coeffs(C, S, L_low)
        grid_low = make_grid_from_coeffs(C_low, S_low)
        
        np.savez_compressed(
            f"{output_dir}/{body_name}_grav_L{L_low}.npz",
            grid=grid_low,
            lmax=L_low
        )
        print(f"  Saved: {grid_low.shape}, {grid_low.nbytes / 1e6:.2f} MB")
    
    # Save raw coefficients for flexibility
    print(f"\nSaving raw coefficients...")
    np.savez_compressed(
        f"{output_dir}/{body_name}_coeffs_raw.npz",
        C=C,
        S=S,
        lmax=max_degree
    )
    
    print(f"\n✓ {body_name.upper()} processing complete!")


if __name__ == "__main__":
    print("="*80)
    print("GRAVITY DATA PREPROCESSING")
    print("Optimized for L=200 training")
    print("="*80)
    
    # Configuration
    MAX_DEGREE = 200
    LOW_DEGREES = [25, 50, 100]
    
    # Process Moon data (for training)
    # Replace this path with your actual Moon data file
    moon_file = "data/train/moon_large/jggrx_1800f_sha.tab"
    
    if os.path.exists(moon_file):
        process_body(
            body_name="moon",
            sha_file_path=moon_file,
            max_degree=MAX_DEGREE,
            low_degrees=LOW_DEGREES
        )
    else:
        print(f"\n⚠ Warning: Moon file not found: {moon_file}")
        print("Skipping Moon processing...")
    
    # Process Mercury data (for application)
    # You need to download Mercury gravity data from NASA PDS
    # Example: HgM007 or HgM008 from MESSENGER mission
    mercury_file = "data/train/mercury/ggmes_100v08_sha.tab"  # ← Update this path!
    
    if os.path.exists(mercury_file):
        process_body(
            body_name="mercury",
            sha_file_path=mercury_file,
            max_degree=100,  # Mercury only goes to ~100 degrees
            low_degrees=LOW_DEGREES
        )
    else:
        print(f"\n⚠ Warning: Mercury file not found: {mercury_file}")
        print("Please download Mercury gravity data from NASA PDS:")
        print("  https://pds-geosciences.wustl.edu/messenger/mess-h-rss_mla-5-sdp-v1/messrs_1001/data/")
        print("  Look for: hgm007_sha.tab or hgm008_sha.tab")
        print("\nSkipping Mercury processing...")
    
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    
    # List all generated files
    processed_dir = "data/processed"
    if os.path.exists(processed_dir):
        files = sorted(os.listdir(processed_dir))
        if files:
            print("\nGenerated files:")
            for f in files:
                filepath = os.path.join(processed_dir, f)
                size_mb = os.path.getsize(filepath) / 1e6
                print(f"  ✓ {f:40s} ({size_mb:6.2f} MB)")
        else:
            print("\nNo files generated - check paths above")
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*80)