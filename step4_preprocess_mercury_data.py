import numpy as np
import os
from PIL import Image
import argparse

Image.MAX_IMAGE_PIXELS = None

TARGET_SHAPE = (360, 720)
DEFAULT_OUTPUT_DIR = "data/processed"
DEFAULT_NORTH_PATH = "data/train/mercury_dem/mercury_north_pole.tiff"
DEFAULT_SOUTH_PATH = "data/train/mercury_dem/mercury_south_pole.tiff"


def load_and_resize_tiff(filepath, target_width):
    """Load TIFF file and resize while preserving floating point precision"""
    print(f"Loading: {filepath}...")

    with Image.open(filepath) as img:
        print(f"  - Original Mode: {img.mode}")

        if img.mode == 'RGB':
            print("  - WARNING: RGB image detected. Converting to float (lossy).")
            img = img.convert('F')

        scale_factor = target_width / img.size[0]
        new_height = int(img.size[1] * scale_factor)

        resized = img.resize((target_width, new_height), Image.Resampling.BICUBIC)
        return np.array(resized)


def stitch_hemispheres(north_array, south_array, target_shape):
    """Vertically stack and resize hemisphere arrays to target shape"""
    stitched = np.vstack([north_array, south_array])

    if stitched.shape != target_shape:
        print(f"Resizing from {stitched.shape} to {target_shape}...")
        img = Image.fromarray(stitched)
        img = img.resize((target_shape[1], target_shape[0]), Image.Resampling.BICUBIC)
        stitched = np.array(img)

    return stitched


def process_mercury_dem(north_path, south_path, output_dir, target_width=720, target_shape=TARGET_SHAPE):
    """Process Mercury DEM by loading, stitching, and saving hemisphere data"""
    os.makedirs(output_dir, exist_ok=True)

    print("--- Processing Mercury DEM ---")

    if not os.path.exists(north_path):
        raise FileNotFoundError(f"North pole file not found: {north_path}")
    if not os.path.exists(south_path):
        raise FileNotFoundError(f"South pole file not found: {south_path}")

    north_arr = load_and_resize_tiff(north_path, target_width)
    south_arr = load_and_resize_tiff(south_path, target_width)

    global_dem = stitch_hemispheres(north_arr, south_arr, target_shape)

    output_path = os.path.join(output_dir, f"mercury_dem_{target_shape[1]}x{target_shape[0]}.npy")
    np.save(output_path, global_dem)

    print(f"Saved DEM: {output_path}")
    print(f"  Range: [{global_dem.min():.2f}, {global_dem.max():.2f}]")
    print(f"  Shape: {global_dem.shape}")

    return global_dem


def main():
    process_mercury_dem(
        north_path=DEFAULT_NORTH_PATH,
        south_path=DEFAULT_SOUTH_PATH,
        output_dir=DEFAULT_OUTPUT_DIR
    )
    
if __name__ == "__main__":
    main()