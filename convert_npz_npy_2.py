"""
Convert Mercury spherical harmonic coefficients to gravity grids
Generates the L25 and L50 gravity maps needed for training/reconstruction
"""
import numpy as np
import os

def compute_legendre_polynomials(lmax, theta):
    """
    Compute fully normalized associated Legendre polynomials
    P_lm(cos(theta)) for all l <= lmax and m <= l
    
    Returns: dict with keys (l,m) -> array of values at each theta
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    P = {}
    
    # P_00
    P[(0, 0)] = np.ones_like(theta)
    
    # P_10
    if lmax >= 1:
        P[(1, 0)] = np.sqrt(3) * cos_theta
    
    # P_11
    if lmax >= 1:
        P[(1, 1)] = np.sqrt(3) * sin_theta
    
    # Recursion for higher degrees
    for l in range(2, lmax + 1):
        # Diagonal term P_ll
        P[(l, l)] = np.sqrt((2*l + 1) / (2*l)) * sin_theta * P[(l-1, l-1)]
        
        # P_l,l-1
        P[(l, l-1)] = np.sqrt(2*l + 1) * cos_theta * P[(l-1, l-1)]
        
        # General recursion for m < l-1
        for m in range(l-1):
            a_lm = np.sqrt(((2*l + 1) * (2*l - 1)) / ((l - m) * (l + m)))
            b_lm = np.sqrt(((2*l + 1) * (l + m - 1) * (l - m - 1)) / 
                          ((l - m) * (l + m) * (2*l - 3)))
            
            P[(l, m)] = a_lm * cos_theta * P[(l-1, m)]
            if l >= 2:
                P[(l, m)] -= b_lm * P[(l-2, m)]
    
    return P

def coeffs_to_gravity_grid(C_lm, S_lm, GM, R_ref, lmax_use, nlat, nlon):
    """
    Convert spherical harmonic coefficients to gravity anomaly grid
    
    Args:
        C_lm, S_lm: Coefficient arrays (lmax+1, lmax+1)
        GM: Gravitational parameter (m^3/s^2)
        R_ref: Reference radius (m)
        lmax_use: Maximum degree to use (can be less than full lmax)
        nlat, nlon: Output grid size
    
    Returns:
        gravity_grid: (nlat, nlon) gravity anomalies in mGal
    """
    print(f"  Computing gravity grid at L={lmax_use}...")
    print(f"    Grid size: {nlat} x {nlon}")
    
    # Create lat/lon grids
    lats = np.linspace(90, -90, nlat)  # North to South
    lons = np.linspace(-180, 180, nlon, endpoint=False)
    
    # Convert to radians
    theta = np.radians(90 - lats)  # Colatitude
    phi = np.radians(lons)
    
    # Initialize output
    gravity = np.zeros((nlat, nlon))
    
    # Compute Legendre polynomials for all latitudes
    print(f"    Computing Legendre polynomials...")
    P = compute_legendre_polynomials(lmax_use, theta)
    
    # Sum over all harmonics
    print(f"    Summing harmonics (L=0 to {lmax_use})...")
    for l in range(lmax_use + 1):
        if l % 20 == 0:
            print(f"      Processing degree {l}...")
        
        for m in range(l + 1):
            # Get Legendre polynomial
            P_lm = P[(l, m)]
            
            # Get coefficients (handle potential out of bounds)
            if l < C_lm.shape[0] and m < C_lm.shape[1]:
                C = C_lm[l, m]
                S = S_lm[l, m] if m > 0 else 0
            else:
                continue
            
            # Skip if both coefficients are zero
            if C == 0 and S == 0:
                continue
            
            # Compute spherical harmonic
            # Y_lm = P_lm * (C * cos(m*phi) + S * sin(m*phi))
            for j, lon_rad in enumerate(phi):
                cos_term = C * np.cos(m * lon_rad)
                sin_term = S * np.sin(m * lon_rad)
                
                # Add contribution to all latitudes at this longitude
                gravity[:, j] += P_lm * (cos_term + sin_term) * (l + 1)
    
    # Apply gravity formula: g = (GM/R^2) * sum
    # Convert to mGal (1 mGal = 10^-5 m/s^2)
    gravity = (GM / (R_ref ** 2)) * gravity * 1e5
    
    print(f"    Gravity range: [{gravity.min():.2f}, {gravity.max():.2f}] mGal")
    
    return gravity

def main():
    print("="*80)
    print("MERCURY SPHERICAL HARMONIC COEFFICIENTS → GRAVITY GRIDS")
    print("="*80)
    
    # Load coefficients
    coeff_path = 'data/train/mercury/mercury_coeffs.npz'
    print(f"\nLoading coefficients from: {coeff_path}")
    
    if not os.path.exists(coeff_path):
        print(f"ERROR: File not found: {coeff_path}")
        return
    
    data = np.load(coeff_path)
    C_lm = data['C_lm']
    S_lm = data['S_lm']
    GM = data['GM']
    R_ref = data['R_ref']
    l_max = int(data['l_max'])
    
    print(f"  Coefficients loaded:")
    print(f"    Max degree: {l_max}")
    print(f"    C_lm shape: {C_lm.shape}")
    print(f"    S_lm shape: {S_lm.shape}")
    print(f"    GM: {GM:.6e} m^3/s^2")
    print(f"    R_ref: {R_ref:.2f} m")
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate L25 (low-resolution for input)
    print("\n" + "-"*80)
    print("GENERATING L25 GRAVITY (Low Resolution Input)")
    print("-"*80)
    
    # For L25, use 50 lats x 100 lons (about 3.6° resolution)
    nlat_25 = 50
    nlon_25 = 100
    
    grav_L25 = coeffs_to_gravity_grid(
        C_lm, S_lm, GM, R_ref, 
        lmax_use=25,
        nlat=nlat_25, 
        nlon=nlon_25
    )
    
    np.save('data/processed/mercury_grav_L25.npy', grav_L25)
    print(f"  ✓ Saved: data/processed/mercury_grav_L25.npy")
    print(f"    Shape: {grav_L25.shape}")
    
    # Generate L50 (high-resolution ground truth for North)
    print("\n" + "-"*80)
    print("GENERATING L50 GRAVITY (High Resolution Ground Truth)")
    print("-"*80)
    
    # For L50, use 100 lats x 200 lons (about 1.8° resolution)
    nlat_50 = 100
    nlon_50 = 200
    
    grav_L50 = coeffs_to_gravity_grid(
        C_lm, S_lm, GM, R_ref,
        lmax_use=50,
        nlat=nlat_50,
        nlon=nlon_50
    )
    
    np.save('data/processed/mercury_grav_L50.npy', grav_L50)
    print(f"  ✓ Saved: data/processed/mercury_grav_L50.npy")
    print(f"    Shape: {grav_L50.shape}")
    
    # Generate L100 (high-resolution ground truth)
    print("\n" + "-"*80)
    print("GENERATING L100 GRAVITY (High Resolution Ground Truth)")
    print("-"*80)
    
    # For L100, use 200 lats x 400 lons (about 0.9° resolution)
    nlat_100 = 200
    nlon_100 = 400
    
    grav_L100 = coeffs_to_gravity_grid(
        C_lm, S_lm, GM, R_ref,
        lmax_use=min(100, l_max),  # Use full L100 if available
        nlat=nlat_100,
        nlon=nlon_100
    )
    
    np.save('data/processed/mercury_grav_L100.npy', grav_L100)
    print(f"  ✓ Saved: data/processed/mercury_grav_L100.npy")
    print(f"    Shape: {grav_L100.shape}")
    
    # Check DEM file
    print("\n" + "-"*80)
    print("CHECKING DEM FILE")
    print("-"*80)
    
    # Your reconstruction expects mercury_dem_720x360.npy
    # But you have mercury_dem_L50.npy with shape (100, 201)
    dem_files = [
        ('data/processed/mercury_dem_L50.npy', (100, 201)),
        ('data/processed/mercury_dem_720x360.npy', None),
    ]
    
    for dem_file, expected_shape in dem_files:
        if os.path.exists(dem_file):
            dem = np.load(dem_file)
            print(f"  ✓ Found: {dem_file}")
            print(f"    Shape: {dem.shape}")
            if expected_shape and dem.shape != expected_shape:
                print(f"    ⚠ Warning: Expected {expected_shape}")
        else:
            print(f"  ✗ Missing: {dem_file}")
    
    # Summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. data/processed/mercury_grav_L25.npy  (50x100) - Low-res input")
    print("  2. data/processed/mercury_grav_L50.npy  (100x200) - Mid-res")
    print("  3. data/processed/mercury_grav_L100.npy (200x400) - High-res ground truth")
    print("\nNext steps:")
    print("  1. Generate/verify DEM at 200x400 resolution")
    print("  2. RETRAIN on Moon: python train_with_dem.py --l_low 25 --l_high 100")
    print("  3. Fine-tune on Mercury: use L25→L100 with mercury_dem_L100.npy")
    print("  4. Reconstruct with L100 for much better detail!")
    print("="*80)

if __name__ == "__main__":
    main()