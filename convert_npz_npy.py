import numpy as np

# Convert your gravity files from .npz to .npy
for L in [25, 50, 200]:
    data = np.load(f'data/processed/moon_grav_L{L}.npz')
    np.save(f'data/processed/moon_grav_L{L}.npy', data['grid'])
    print(f"✓ Converted moon_grav_L{L}.npz → moon_grav_L{L}.npy")

# Convert MERCURY gravity files from .npz to .npy
# Note: We only need L25 and L50 for the Mercury fine-tuning step
for L in [25, 50]:
    try:
        # Load the .npz file
        data = np.load(f'data/processed/mercury_grav_L{L}.npz')
        
        # Save the 'grid' array inside it as a .npy file
        np.save(f'data/processed/mercury_grav_L{L}.npy', data['grid'])
        
        print(f"✓ Converted mercury_grav_L{L}.npz → mercury_grav_L{L}.npy")
    except FileNotFoundError:
        print(f"X File data/processed/mercury_grav_L{L}.npz not found.")

# Convert MERCURY DEM file
# (Assuming your DEM .npz is named mercury_dem_720x360.npz based on your previous command)
try:
    dem_data = np.load('data/processed/mercury_dem_720x360.npz')
    np.save('data/processed/mercury_dem_720x360.npy', dem_data['grid'])
    print("✓ Converted mercury_dem_720x360.npz → mercury_dem_720x360.npy")
except FileNotFoundError:
    print("X File mercury_dem_720x360.npz not found (check the exact filename in data/processed/)")

# Load the L50 DEM archive
data = np.load('data/processed/mercury_dem_L50.npz')

# Save it as a .npy file
np.save('data/processed/mercury_dem_L50.npy', data['grid'])

print("✓ Created data/processed/mercury_dem_L50.npy")

# Load the archive
data = np.load('data/processed/mercury_dem_L200.npz')

# Save as .npy
np.save('data/processed/mercury_dem_L200.npy', data['grid'])
print("✓ Created data/processed/mercury_dem_L200.npy")