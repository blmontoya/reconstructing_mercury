"""
DEM Data Preprocessing Pipeline
Process Lunar LOLA and Mercury MDIS DEM data for gravity field reconstruction
"""
import numpy as np
from scipy.ndimage import zoom
import os
import matplotlib.pyplot as plt


def load_tif_dem(tif_path):
    """
    Load DEM data from GeoTIFF file using multiple backends

    Args:
        tif_path: Path to .tif DEM file

    Returns:
        dem_array: 2D numpy array of elevation data
        metadata: Dictionary with geospatial metadata
    """
    print(f"\nLoading DEM from {tif_path}...")

    # Try rasterio first (more robust for GeoTIFFs)
    try:
        import rasterio
        from rasterio.enums import Resampling
        with rasterio.open(tif_path) as dataset:
            # Read downsampled to save RAM (factor of 16 = 2880×1440 max)
            dem_array = dataset.read(
                1,
                out_shape=(
                    dataset.height // 16,
                    dataset.width // 16
                ),
                resampling=Resampling.bilinear
            )
            metadata = {
                'width': dataset.width,
                'height': dataset.height,
                'bounds': dataset.bounds,
                'crs': dataset.crs,
                'transform': dataset.transform
            }
        print(f"  Loaded using rasterio")

    except ImportError:
        # Fall back to PIL for basic TIFF reading
        try:
            from PIL import Image
            img = Image.open(tif_path)
            dem_array = np.array(img)
            metadata = {
                'width': img.width,
                'height': img.height
            }
            print(f"  Loaded using PIL")

        except ImportError:
            # Last resort: try GDAL
            try:
                from osgeo import gdal
                dataset = gdal.Open(tif_path)
                if dataset is None:
                    raise FileNotFoundError(f"Could not open {tif_path}")

                band = dataset.GetRasterBand(1)
                dem_array = band.ReadAsArray()
                metadata = {
                    'geotransform': dataset.GetGeoTransform(),
                    'projection': dataset.GetProjection(),
                    'width': dataset.RasterXSize,
                    'height': dataset.RasterYSize,
                    'nodata': band.GetNoDataValue()
                }
                dataset = None
                print(f"  Loaded using GDAL")

            except ImportError:
                raise ImportError("No suitable library found to read GeoTIFF. "
                                "Please install: pip install rasterio (recommended) "
                                "or pip install pillow")

    print(f"  DEM shape: {dem_array.shape}")
    print(f"  Resolution: {metadata['width']} x {metadata['height']}")
    print(f"  Elevation range: [{np.nanmin(dem_array):.2f}, {np.nanmax(dem_array):.2f}] m")

    return dem_array, metadata


def resize_dem_to_match_gravity(dem_array, target_shape, order=1):
    """
    Resize DEM to match the target gravity field resolution

    Args:
        dem_array: Input DEM array
        target_shape: Desired output shape (H, W)
        order: Interpolation order (1=linear, 3=cubic)

    Returns:
        Resized DEM array
    """
    print(f"\nResizing DEM from {dem_array.shape} to {target_shape}...")

    # Handle NaN values by replacing with mean
    dem_clean = dem_array.copy()
    nan_mask = np.isnan(dem_clean)
    if nan_mask.any():
        dem_mean = np.nanmean(dem_clean)
        dem_clean[nan_mask] = dem_mean
        print(f"  Replaced {nan_mask.sum()} NaN values with mean ({dem_mean:.2f})")

    # Calculate zoom factors
    zoom_factors = (
        target_shape[0] / dem_clean.shape[0],
        target_shape[1] / dem_clean.shape[1]
    )

    print(f"  Zoom factors: {zoom_factors[0]:.4f} x {zoom_factors[1]:.4f}")

    # Resize using scipy zoom
    dem_resized = zoom(dem_clean, zoom_factors, order=order)

    print(f"  Output shape: {dem_resized.shape}")
    print(f"  Elevation range: [{dem_resized.min():.2f}, {dem_resized.max():.2f}] m")

    return dem_resized


def normalize_dem(dem_array, method='standard'):
    """
    Normalize DEM data for neural network input

    Args:
        dem_array: Input DEM array
        method: 'standard' (mean=0, std=1) or 'minmax' (range 0-1)

    Returns:
        normalized_dem: Normalized DEM array
        norm_params: Dictionary with normalization parameters
    """
    print(f"\nNormalizing DEM using '{method}' method...")

    if method == 'standard':
        mean = np.mean(dem_array)
        std = np.std(dem_array)
        normalized_dem = (dem_array - mean) / (std + 1e-8)
        norm_params = {'method': 'standard', 'mean': mean, 'std': std}
        print(f"  Mean: {mean:.2f}, Std: {std:.2f}")

    elif method == 'minmax':
        min_val = np.min(dem_array)
        max_val = np.max(dem_array)
        normalized_dem = (dem_array - min_val) / (max_val - min_val + 1e-8)
        norm_params = {'method': 'minmax', 'min': min_val, 'max': max_val}
        print(f"  Min: {min_val:.2f}, Max: {max_val:.2f}")

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    print(f"  Normalized range: [{normalized_dem.min():.4f}, {normalized_dem.max():.4f}]")

    return normalized_dem, norm_params


def process_lunar_dem(
    tif_path='data/train/moon_dem/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif',
    target_shapes=[402, 804, 1440, 2880],  # Corresponding to L25, L50, L100, L200
    output_dir='data/processed',
    visualize=True
):
    """
    Complete pipeline to process Lunar LOLA DEM data

    Args:
        tif_path: Path to Lunar LOLA DEM TIFF file
        target_shapes: List of target resolutions (widths in pixels)
        output_dir: Directory to save processed DEMs
        visualize: Whether to create visualization plots

    Returns:
        Dictionary mapping resolution names to processed DEM arrays
    """
    print("="*80)
    print("LUNAR DEM PROCESSING PIPELINE")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the DEM
    dem_array, metadata = load_tif_dem(tif_path)

    # Process for each target resolution
    processed_dems = {}

    for width in target_shapes:
        print(f"\n{'='*80}")
        print(f"Processing DEM for width={width} pixels")
        print(f"{'='*80}")

        # Assume aspect ratio is 2:1 (longitude:latitude)
        height = width // 2
        target_shape = (height, width)

        # Resize DEM
        dem_resized = resize_dem_to_match_gravity(dem_array, target_shape, order=1)

        # Normalize
        dem_normalized, norm_params = normalize_dem(dem_resized, method='standard')

        # Save processed DEM
        output_name = f'moon_dem_{width}x{height}'
        output_path = os.path.join(output_dir, f'{output_name}.npy')
        np.save(output_path, dem_normalized)
        print(f"\n  Saved to {output_path}")

        # Save normalization parameters
        norm_path = os.path.join(output_dir, f'{output_name}_norm.npz')
        np.savez(norm_path, **norm_params)
        print(f"  Saved normalization params to {norm_path}")

        processed_dems[output_name] = dem_normalized

        # Visualize
        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Original elevation
            im1 = axes[0].imshow(dem_resized, cmap='terrain', aspect='auto')
            axes[0].set_title(f'Lunar DEM {width}x{height} - Elevation (m)',
                            fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Longitude')
            axes[0].set_ylabel('Latitude')
            plt.colorbar(im1, ax=axes[0], label='Elevation (m)')

            # Normalized
            im2 = axes[1].imshow(dem_normalized, cmap='RdBu_r', aspect='auto')
            axes[1].set_title(f'Lunar DEM {width}x{height} - Normalized',
                            fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            plt.colorbar(im2, ax=axes[1], label='Normalized Value')

            plt.tight_layout()
            viz_path = os.path.join(output_dir, f'{output_name}_visualization.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to {viz_path}")
            plt.close()

    print("\n" + "="*80)
    print("DEM PROCESSING COMPLETE")
    print("="*80)
    print(f"Processed {len(processed_dems)} DEM resolutions:")
    for name in processed_dems.keys():
        print(f"  - {name}")

    return processed_dems


def create_aligned_dem_patches(dem_array, patch_size=30, stride=20):
    """
    Extract patches from DEM array (matching the gravity patch extraction)

    Args:
        dem_array: 2D DEM array
        patch_size: Size of patches (default 30x30)
        stride: Stride between patches (default 20)

    Returns:
        Array of DEM patches with shape (N, patch_size, patch_size, 1)
    """
    print(f"\nExtracting {patch_size}x{patch_size} DEM patches with stride={stride}...")

    h, w = dem_array.shape
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = dem_array[i:i+patch_size, j:j+patch_size]

            # Skip patches with NaN values
            if np.isnan(patch).any():
                continue

            patches.append(patch)

    patches = np.array(patches, dtype=np.float32)[..., np.newaxis]
    print(f"  Extracted {len(patches)} patches")

    return patches


def calculate_dem_gravity_ssim(dem_array, gravity_array):
    """
    Calculate SSIM between DEM and gravity field to verify spatial similarity
    (As mentioned in the paper - SSIM > 0.96 for Moon)

    Args:
        dem_array: DEM data (normalized)
        gravity_array: Gravity field data (normalized)

    Returns:
        SSIM value
    """
    from skimage.metrics import structural_similarity as ssim

    # Ensure same shape
    if dem_array.shape != gravity_array.shape:
        print(f"Warning: Shape mismatch - DEM {dem_array.shape}, Gravity {gravity_array.shape}")
        return None

    # Calculate SSIM
    ssim_value = ssim(dem_array, gravity_array, data_range=dem_array.max() - dem_array.min())

    print(f"\nDEM-Gravity SSIM: {ssim_value:.4f}")
    if ssim_value > 0.96:
        print("  High spatial similarity confirmed (SSIM > 0.96)")
    else:
        print("  Lower than expected spatial similarity")

    return ssim_value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process DEM data for gravity reconstruction')
    parser.add_argument('--tif_path', type=str,
                       default='data/train/moon_dem/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif',
                       help='Path to input DEM TIFF file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed DEMs')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualization plots')

    args = parser.parse_args()


    # 360x180 = L25, 720x360 = L50, 1440x720 = L100, 2880x1440 = L200
    processed_dems = process_lunar_dem(
        tif_path=args.tif_path,
        target_shapes=[402, 804, 1440, 2880],
        output_dir=args.output_dir,
        visualize=args.visualize
    )
