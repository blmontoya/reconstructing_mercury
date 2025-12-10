import numpy as np
import os
from PIL import Image
import argparse

# Increase PIL limit for large scientific images
Image.MAX_IMAGE_PIXELS = None 

def load_tiff_preserve_precision(filepath, target_width=720):
    """
    Loads a TIFF and resizes it WITHOUT converting to 8-bit grayscale.
    Preserves floating point values (mGal/Meters).
    """
    print(f"Loading: {filepath}...")
    try:
        img = Image.open(filepath)
        
        # Check if it's already a suitable mode (F=Float32, I=Int32)
        # If it's RGB (visual map), we must convert, but preferably to 'F' not 'L'
        print(f"  - Original Mode: {img.mode}")
        
        if img.mode == 'RGB':
            print("  ! WARNING: Input is an RGB image, not raw data. Converting to grayscale (lossy).")
            img = img.convert('F') # Convert to Float, not 8-bit L
        
        # Calculate height to keep aspect ratio
        w_percent = (target_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        
        # Resize using a filter that handles floats well
        img = img.resize((target_width, h_size), Image.Resampling.BICUBIC)
        
        return np.array(img)
        
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return None

def main():
    # --- CONFIGURATION ---
    north_path = r"data/train/mercury_dem/mercury_north_pole.tiff"
    south_path = r"data/train/mercury_dem/mercury_south_pole.tiff"
    output_dir = "data/processed"
    TARGET_WIDTH = 720
    # ---------------------

    os.makedirs(output_dir, exist_ok=True)

    # 1. LOAD AND STITCH DEM
    print("--- Processing DEM ---")
    if not os.path.exists(north_path) or not os.path.exists(south_path):
        print("ERROR: Could not find North or South TIFF files.")
        return

    north_arr = load_tiff_preserve_precision(north_path, TARGET_WIDTH)
    south_arr = load_tiff_preserve_precision(south_path, TARGET_WIDTH)
    
    # Validation
    if north_arr is None or south_arr is None:
        print("Failed to load images.")
        return

    # Stitch
    global_dem = np.vstack([north_arr, south_arr])
    
    # Force exact 360x720 shape
    if global_dem.shape != (360, 720):
        print(f"Resizing merged map from {global_dem.shape} to (360, 720)...")
        # Convert back to PIL for final resize if stacking messed up dimensions
        img = Image.fromarray(global_dem)
        img = img.resize((720, 360), Image.Resampling.BICUBIC)
        global_dem = np.array(img)

    # Save DEM
    np.save(os.path.join(output_dir, "mercury_dem_720x360.npy"), global_dem)
    print(f"SAVED DEM: Range [{np.min(global_dem):.2f}, {np.max(global_dem):.2f}]")

    # 2. GENERATE GRAVITY (Synthetic for Training)
    print("\n--- Processing Gravity ---")
    from scipy.ndimage import gaussian_filter
    
    # High Res (L50) - Stays relatively sharp
    grav_high = gaussian_filter(global_dem, sigma=2)
    # Multiplier: 25 gives us a range roughly around -80 to +80
    grav_high = (grav_high - np.mean(grav_high)) / (np.std(grav_high) + 1e-8) * 25
    
    np.save(os.path.join(output_dir, "mercury_grav_L50.npy"), grav_high)
    print(f"SAVED L50: Range [{np.min(grav_high):.2f}, {np.max(grav_high):.2f}]")

    # Low Res (L25) - MAKE THIS BLOBBY
    # CHANGE: Increased sigma from 5 to 16. 
    # This blurs out the craters and leaves only the large "blobs" like the real map.
    grav_low = gaussian_filter(grav_high, sigma=16) 
    
    np.save(os.path.join(output_dir, "mercury_grav_L25.npy"), grav_low)
    print(f"SAVED L25: Range [{np.min(grav_low):.2f}, {np.max(grav_low):.2f}]")

if __name__ == "__main__":
    main()