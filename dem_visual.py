# Visualize with better colormap
import numpy as np
import matplotlib.pyplot as plt

moon = np.load('data/processed/moon_dem_L200.npz')['grid']
mercury = np.load('data/processed/mercury_dem_L200.npz')['grid']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Use 'gray' colormap to see craters better (like real DEM imagery)
im1 = ax1.imshow(moon, cmap='gray', vmin=-6000, vmax=6000)
ax1.set_title(f'Synthetic Moon DEM\nRange: [{moon.min():.0f}, {moon.max():.0f}]m')
plt.colorbar(im1, ax=ax1, label='Elevation (m)')

im2 = ax2.imshow(mercury, cmap='gray', vmin=-3000, vmax=3000)
ax2.set_title(f'Synthetic Mercury DEM\nRange: [{mercury.min():.0f}, {mercury.max():.0f}]m')
plt.colorbar(im2, ax=ax2, label='Elevation (m)')

plt.tight_layout()
plt.savefig('synthetic_dems_improved.png', dpi=150)
print('Saved synthetic_dems_improved.png')