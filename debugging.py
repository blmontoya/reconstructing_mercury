import numpy as np

# Load your training data
low = np.load('data/processed/moon_grav_L25.npz')['grid']
high = np.load('data/processed/moon_grav_L200.npz')['grid']

print(f"L=25 shape: {low.shape}")
print(f"L=200 shape: {high.shape}")
print(f"L=25 std: {np.std(low):.4f}")
print(f"L=200 std: {np.std(high):.4f}")

# Check if L=200 actually has more detail
print(f"\nL=25 value range: [{np.min(low):.2f}, {np.max(low):.2f}]")
print(f"L=200 value range: [{np.min(high):.2f}, {np.max(high):.2f}]")

# Add to check_data.py
coeffs = np.load('data/processed/moon_coeffs_raw.npz')
C = coeffs['C']
print(f"\nCoefficient matrix shape: {C.shape}")
print(f"Max degree in source data: {C.shape[0] - 1}")

coeffs = np.load('data/processed/moon_coeffs_raw.npz')
C = coeffs['C']

# Check coefficient magnitudes at different degrees
print("\nCoefficient magnitudes by degree:")
for L in [25, 50, 100, 200, 400, 600]:
    if L < C.shape[0]:
        # Get coefficients at degree L
        c_at_L = C[L, :min(L+1, C.shape[1])]
        mag = np.sqrt(np.sum(c_at_L**2))
        print(f"  L={L:3d}: magnitude = {mag:.2e}")