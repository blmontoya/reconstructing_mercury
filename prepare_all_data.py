import numpy as np
import pandas as pd
import pyshtools as pysh
import os
import glob
from scipy.ndimage import zoom, gaussian_filter

# --- CONFIGURATION ---
DATA_DIR = os.path.join('data', 'train', 'mercury')
OUTPUT_DIR = os.path.join('data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target Resolution: 360x720 (0.5 degree per pixel)
TARGET_SHAPE = (360, 720) 

def load_sh_coefficients(filepath):
    """Parses PDS .tab file into Spherical Harmonic Coefficients."""
    print(f"   Parsing: {os.path.basename(filepath)}")
    try:
        # Determine separator (comma or space)
        with open(filepath, 'r') as f:
            first_line = f.readline()
        sep = ',' if ',' in first_line else r'\s+'
        skip = 1 if ',' in first_line else 0

        # Read Data
        df = pd.read_csv(filepath, sep=sep, skiprows=skip, header=None, comment='#')
        
        # Standardize columns: Degree(l), Order(m), C, S...
        # We grab the first 4 columns regardless of name
        df = df.iloc[:, [0, 1, 2, 3]]
        df.columns = ['l', 'm', 'C', 'S']

        # Construct Matrix
        L_max = int(df['l'].max())
        coeffs = np.zeros((2, L_max + 1, L_max + 1))
        
        for row in df.itertuples():
            l, m = int(row.l), int(row.m)
            coeffs[0, l, m] = row.C
            coeffs[1, l, m] = row.S
            
        return pysh.SHCoeffs.from_array(coeffs)

    except Exception as e:
        print(f"   ❌ Error loading {filepath}: {e}")
        return None

def sh_to_grid(coeffs, lmax=None):
    """Converts coefficients to a spatial grid."""
    if lmax:
        print(f"   Math: Truncating to Degree {lmax}...")
        grid = coeffs.expand(grid='DH2', lmax=lmax)
    else:
        print(f"   Math: Expanding to Max Degree {coeffs.lmax}...")
        grid = coeffs.expand(grid='DH2')
    return grid.data

def resize_grid(grid, shape):
    """Resizes grid to target shape (360x720)."""
    zoom_h = shape[0] / grid.shape[0]
    zoom_w = shape[1] / grid.shape[1]
    return zoom(grid, (zoom_h, zoom_w), order=1)

def main():
    print("="*60)
    print("MERCURY DATA PREPARATION PIPELINE")
    print("="*60)

    # ---------------------------------------------------------
    # 1. PROCESS GRAVITY (GGMES)
    # ---------------------------------------------------------
    print("\n1. LOCATING GRAVITY DATA...")
    grav_files = glob.glob(os.path.join(DATA_DIR, "*ggmes*sha.tab"))
    
    if grav_files:
        grav_path = grav_files[0]
        grav_coeffs = load_sh_coefficients(grav_path)

        if grav_coeffs:
            # A. Generate High-Res Target (L50)
            print("   Generating High-Res Target (L50)...")
            # We truncate to L50 for the "Truth" to match the resolution we can realistically reconstruct
            grid_raw = sh_to_grid(grav_coeffs, lmax=50) 
            grav_high = resize_grid(grid_raw, TARGET_SHAPE)
            
            # Save L50
            np.save(os.path.join(OUTPUT_DIR, "mercury_grav_L50.npy"), grav_high)
            print("   ✅ Saved: mercury_grav_L50.npy")

            # B. Generate Low-Res Input (L25) - The input for the AI
            print("   Generating Low-Res Input (L25)...")
            # We strictly truncate math to L25 (simulating lower resolution sensor)
            grid_low_raw = sh_to_grid(grav_coeffs, lmax=25)
            grav_low = resize_grid(grid_low_raw, TARGET_SHAPE)
            
            # Save L25
            np.save(os.path.join(OUTPUT_DIR, "mercury_grav_L25.npy"), grav_low)
            print("   ✅ Saved: mercury_grav_L25.npy")
    else:
        print("❌ ERROR: No GGMES (Gravity) file found!")

    # ---------------------------------------------------------
    # 2. PROCESS TOPOGRAPHY (GTMES)
    # ---------------------------------------------------------
    print("\n2. LOCATING TOPOGRAPHY (DEM) DATA...")
    dem_files = glob.glob(os.path.join(DATA_DIR, "*gtmes*sha.tab")) + \
                glob.glob(os.path.join(DATA_DIR, "*GTMES*sha.tab"))

    if dem_files:
        dem_path = dem_files[0]
        dem_coeffs = load_sh_coefficients(dem_path)

        if dem_coeffs:
            # Generate DEM (Resolution matches Gravity Target L50)
            grid_dem_raw = sh_to_grid(dem_coeffs, lmax=50)
            dem_high = resize_grid(grid_dem_raw, TARGET_SHAPE)
            
            # Save DEM
            np.save(os.path.join(OUTPUT_DIR, "mercury_dem_L50.npy"), dem_high)
            print("   ✅ Saved: mercury_dem_L50.npy")
    else:
        print("❌ WARNING: No GTMES (Topography) file found.")
        print("   The training script will crash without this.")
        print("   Generating a TEMPORARY placeholder using gravity data...")
        # Create a fake DEM from gravity just so code runs (NOT SCIENTIFICALLY ACCURATE)
        if 'grav_high' in locals():
            np.save(os.path.join(OUTPUT_DIR, "mercury_dem_L50.npy"), grav_high)
            print("   ⚠️ SAVED PLACEHOLDER DEM (Copy of Gravity). Replace with real data later!")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE.")
    print("Ready for: python train_mercury_finetuning.py")
    print("="*60)

if __name__ == "__main__":
    main()