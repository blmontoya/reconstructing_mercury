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
    try:
        import rasterio
        from rasterio.enums import Resampling
        with rasterio.open(tif_path) as dataset:
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

    except ImportError:
        try:
            from PIL import Image
            img = Image.open(tif_path)
            dem_array = np.array(img)
            metadata = {
                'width': img.width,
                'height': img.height
            }

        except ImportError:
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

            except ImportError:
                raise ImportError("No suitable library found to read GeoTIFF. "
                                "Please install: pip install rasterio (recommended) "
                                "or pip install pillow")

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
    dem_clean = dem_array.copy()
    nan_mask = np.isnan(dem_clean)
    if nan_mask.any():
        dem_mean = np.nanmean(dem_clean)
        dem_clean[nan_mask] = dem_mean

    zoom_factors = (
        target_shape[0] / dem_clean.shape[0],
        target_shape[1] / dem_clean.shape[1]
    )

    dem_resized = zoom(dem_clean, zoom_factors, order=order)

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
    if method == 'standard':
        mean = np.mean(dem_array)
        std = np.std(dem_array)
        normalized_dem = (dem_array - mean) / (std + 1e-8)
        norm_params = {'method': 'standard', 'mean': mean, 'std': std}

    elif method == 'minmax':
        min_val = np.min(dem_array)
        max_val = np.max(dem_array)
        normalized_dem = (dem_array - min_val) / (max_val - min_val + 1e-8)
        norm_params = {'method': 'minmax', 'min': min_val, 'max': max_val}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_dem, norm_params


def process_lunar_dem(
    tif_path='data/train/moon_dem/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif',
    target_shapes=[402, 804, 1440, 2880],
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
    os.makedirs(output_dir, exist_ok=True)

    dem_array, metadata = load_tif_dem(tif_path)

    processed_dems = {}

    for width in target_shapes:
        height = width // 2
        target_shape = (height, width)

        dem_resized = resize_dem_to_match_gravity(dem_array, target_shape, order=1)
        dem_normalized, norm_params = normalize_dem(dem_resized, method='standard')

        output_name = f'moon_dem_{width}x{height}'
        output_path = os.path.join(output_dir, f'{output_name}.npy')
        np.save(output_path, dem_normalized)

        norm_path = os.path.join(output_dir, f'{output_name}_norm.npz')
        np.savez(norm_path, **norm_params)

        processed_dems[output_name] = dem_normalized

        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            im1 = axes[0].imshow(dem_resized, cmap='terrain', aspect='auto')
            axes[0].set_title(f'Lunar DEM {width}x{height} - Elevation (m)',
                            fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Longitude')
            axes[0].set_ylabel('Latitude')
            plt.colorbar(im1, ax=axes[0], label='Elevation (m)')

            im2 = axes[1].imshow(dem_normalized, cmap='RdBu_r', aspect='auto')
            axes[1].set_title(f'Lunar DEM {width}x{height} - Normalized',
                            fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            plt.colorbar(im2, ax=axes[1], label='Normalized Value')

            plt.tight_layout()
            viz_path = os.path.join(output_dir, f'{output_name}_visualization.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

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
    h, w = dem_array.shape
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = dem_array[i:i+patch_size, j:j+patch_size]

            if np.isnan(patch).any():
                continue

            patches.append(patch)

    patches = np.array(patches, dtype=np.float32)[..., np.newaxis]

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

    if dem_array.shape != gravity_array.shape:
        return None

    ssim_value = ssim(dem_array, gravity_array, data_range=dem_array.max() - dem_array.min())

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
    
    # --- ADD THIS NEW ARGUMENT ---
    parser.add_argument('--target_shapes', type=int, nargs='+', 
                        default=[402, 804, 1440, 2880],
                        help='List of target widths (pixels)')

    args = parser.parse_args()

    # Pass the command line args into the function
    processed_dems = process_lunar_dem(
        tif_path=args.tif_path,
        target_shapes=args.target_shapes,  # <--- Now uses the input argument
        output_dir=args.output_dir,
        visualize=args.visualize
    )