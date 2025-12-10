import numpy as np
from scipy.ndimage import zoom
import os

# ADJUST THIS PATH to point to your raw high-res DEM file
RAW_DEM_PATH = 'data/raw/Mercury_DEM.npy' 
OUTPUT_PATH = 'data/processed/mercury_dem_2880x1440.npy'

# Target size matching your Moon experiment
TARGET_SHAPE = (1440, 2880) 

if os.path.exists(RAW_DEM_PATH):
    print(f"Loading raw DEM from {RAW_DEM_PATH}...")
    raw_dem = np.load(RAW_DEM_PATH)

    print(f"Resizing from {raw_dem.shape} to {TARGET_SHAPE}...")
    # Calculate zoom to hit exactly 1440x2880
    zoom_factors = (TARGET_SHAPE[0] / raw_dem.shape[0], 
                    TARGET_SHAPE[1] / raw_dem.shape[1])

    high_res_dem = zoom(raw_dem, zoom_factors, order=1)

    # Normalize (Standard Score)
    mean = np.mean(high_res_dem)
    std = np.std(high_res_dem)
    high_res_dem = (high_res_dem - mean) / (std + 1e-8)

    np.save(OUTPUT_PATH, high_res_dem)
    print(f"✓ Saved sharp DEM to {OUTPUT_PATH}")
else:
    print(f"X Error: Could not find raw file at {RAW_DEM_PATH}")