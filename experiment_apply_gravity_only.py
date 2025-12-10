"""
Apply trained Moon model to Mercury data
Loads the best saved model and generates Mercury reconstruction visualization

Usage:
    python apply_to_mercury.py
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom

# Import custom layers for model loading
from moon_model import GravityReconstructionNetwork, DenseBlock

# Import evaluation metrics (optional - for validation if you have ground truth)
try:
    from metrics import pearson_correlation, calculate_ssim, calculate_rmse, compare_power_spectra
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: metrics.py not found. Skipping evaluation metrics.")
    METRICS_AVAILABLE = False


def create_patches_from_grid(grid, patch_size=30, stride=15):
    """
    Extract overlapping patches from grid for prediction
    Uses padding to ensure full coverage including edges
    """
    h, w = grid.shape
    
    # Calculate padding needed to ensure full coverage
    pad_h = (patch_size - (h % stride)) % stride if h % stride != 0 else 0
    pad_w = (patch_size - (w % stride)) % stride if w % stride != 0 else 0
    
    # Pad the grid with edge values (better than zeros for gravity fields)
    if pad_h > 0 or pad_w > 0:
        grid_padded = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='edge')
        print(f"      Padded grid from {grid.shape} to {grid_padded.shape}")
    else:
        grid_padded = grid
    
    h_padded, w_padded = grid_padded.shape
    patches = []
    positions = []
    
    # Extract patches with overlap to cover entire grid
    for i in range(0, h_padded - patch_size + 1, stride):
        for j in range(0, w_padded - patch_size + 1, stride):
            patch = grid_padded[i:i+patch_size, j:j+patch_size]
            
            if not np.isnan(patch).any():
                patches.append(patch)
                positions.append((i, j))
    
    # Also add edge patches if needed
    # Right edge
    if (w_padded - patch_size) % stride != 0:
        for i in range(0, h_padded - patch_size + 1, stride):
            j = w_padded - patch_size
            patch = grid_padded[i:i+patch_size, j:j+patch_size]
            if not np.isnan(patch).any():
                patches.append(patch)
                positions.append((i, j))
    
    # Bottom edge
    if (h_padded - patch_size) % stride != 0:
        for j in range(0, w_padded - patch_size + 1, stride):
            i = h_padded - patch_size
            patch = grid_padded[i:i+patch_size, j:j+patch_size]
            if not np.isnan(patch).any():
                patches.append(patch)
                positions.append((i, j))
    
    # Bottom-right corner
    if (h_padded - patch_size) % stride != 0 and (w_padded - patch_size) % stride != 0:
        i = h_padded - patch_size
        j = w_padded - patch_size
        patch = grid_padded[i:i+patch_size, j:j+patch_size]
        if not np.isnan(patch).any():
            patches.append(patch)
            positions.append((i, j))
    
    patches = np.array(patches, dtype=np.float32)[..., np.newaxis]
    
    # Store original shape and padding for reconstruction
    metadata = {
        'original_shape': grid.shape,
        'padded_shape': grid_padded.shape,
        'pad_h': pad_h,
        'pad_w': pad_w
    }
    
    return patches, positions, metadata


def reconstruct_from_patches(patches, positions, metadata, patch_size=30):
    """
    Reconstruct full grid from overlapping patches using averaging
    Handles padding and returns to original dimensions
    """
    padded_shape = metadata['padded_shape']
    original_shape = metadata['original_shape']
    h_padded, w_padded = padded_shape
    patches = patches.squeeze()
    
    reconstructed = np.zeros((h_padded, w_padded), dtype=np.float32)
    counts = np.zeros((h_padded, w_padded), dtype=np.float32)
    
    for patch, (i, j) in zip(patches, positions):
        reconstructed[i:i+patch_size, j:j+patch_size] += patch
        counts[i:i+patch_size, j:j+patch_size] += 1
    
    # Average overlapping regions
    counts[counts == 0] = 1
    reconstructed /= counts
    
    # Remove padding to get back to original shape
    h_orig, w_orig = original_shape
    reconstructed = reconstructed[:h_orig, :w_orig]
    
    print(f"      Reconstructed shape: {reconstructed.shape} (removed padding)")
    
    return reconstructed


def visualize_mercury_reconstruction(grid_low, grid_high_reconstructed, output_path):
    """Create before/after visualization showing the resolution difference"""
    
    # Print dimensions for debugging
    print(f"      Low-res dimensions: {grid_low.shape}")
    print(f"      High-res dimensions: {grid_high_reconstructed.shape}")
    
    # Calculate actual latitude coverage for DH2 grids
    nlat_low = grid_low.shape[0]
    nlat_high = grid_high_reconstructed.shape[0]
    
    # DH2 latitude spacing
    dlat_low = 180.0 / nlat_low
    dlat_high = 180.0 / nlat_high
    
    # Actual latitude range for DH2
    lat_max_low = 90.0 - dlat_low / 2.0
    lat_min_low = -90.0 + dlat_low / 2.0
    
    lat_max_high = 90.0 - dlat_high / 2.0
    lat_min_high = -90.0 + dlat_high / 2.0
    
    print(f"      Low-res latitude range: {lat_max_low:.2f}° to {lat_min_low:.2f}°")
    print(f"      High-res latitude range: {lat_max_high:.2f}° to {lat_min_high:.2f}°")
    
    lon_extent = [0, 360]
    
    # Create TWO visualizations: 1) Full comparison, 2) Zoomed comparison
    
    # ========== FULL COMPARISON ==========
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Color limits
    vmin, vmax = -75, 75

    # Input (Low Resolution)
    im1 = axes[0].imshow(
        grid_low,
        cmap='RdBu_r',
        aspect='auto',
        extent=[lon_extent[0], lon_extent[1], lat_min_low, lat_max_low],
        origin='upper',
        interpolation='bilinear',
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title(f'Input: Mercury L=25\n{grid_low.shape[0]}×{grid_low.shape[1]} = {grid_low.shape[0]*grid_low.shape[1]:,} pixels', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Longitude (degrees)', fontsize=12)
    axes[0].set_ylabel('Latitude (degrees)', fontsize=12)
    axes[0].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
    cbar1 = plt.colorbar(im1, ax=axes[0], label='Gravity (mGal)', fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=10)
    
    # Output (Reconstructed High Resolution)
    im2 = axes[1].imshow(
        grid_high_reconstructed,
        cmap='RdBu_r',
        aspect='auto',
        extent=[lon_extent[0], lon_extent[1], lat_min_high, lat_max_high],
        origin='upper',
        interpolation='bilinear',
        vmin=vmin,
        vmax=vmax
    )
    pixel_ratio = (grid_high_reconstructed.shape[0] * grid_high_reconstructed.shape[1]) / (grid_low.shape[0] * grid_low.shape[1])
    axes[1].set_title(f'Reconstructed: Mercury L=200\n{grid_high_reconstructed.shape[0]}×{grid_high_reconstructed.shape[1]} = {grid_high_reconstructed.shape[0]*grid_high_reconstructed.shape[1]:,} pixels ({pixel_ratio:.1f}x more)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Longitude (degrees)', fontsize=12)
    axes[1].set_ylabel('Latitude (degrees)', fontsize=12)
    axes[1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Gravity (mGal)', fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=10)
    
    plt.suptitle('Mercury Gravity Field Reconstruction\nTrained on Moon (GRAIL) → Applied to Mercury (MESSENGER)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved full comparison to {output_path}")
    plt.close()
    
    # ========== ZOOMED COMPARISON (to show pixel detail) ==========
    # Select a region: latitude 0-40°, longitude 180-240°
    # This shows equatorial region where differences are clearest
    
    from scipy.ndimage import zoom as scipy_zoom
    
    # Define zoom region (indices)
    lat_zoom = [0, 40]  # degrees
    lon_zoom = [180, 240]  # degrees
    
    # Convert to array indices
    lat_idx_low_start = int((90 - lat_zoom[1]) / (180 / nlat_low))
    lat_idx_low_end = int((90 - lat_zoom[0]) / (180 / nlat_low))
    lon_idx_low_start = int(lon_zoom[0] / (360 / grid_low.shape[1]))
    lon_idx_low_end = int(lon_zoom[1] / (360 / grid_low.shape[1]))
    
    lat_idx_high_start = int((90 - lat_zoom[1]) / (180 / nlat_high))
    lat_idx_high_end = int((90 - lat_zoom[0]) / (180 / nlat_high))
    lon_idx_high_start = int(lon_zoom[0] / (360 / grid_high_reconstructed.shape[1]))
    lon_idx_high_end = int(lon_zoom[1] / (360 / grid_high_reconstructed.shape[1]))
    
    # Extract regions
    zoom_low = grid_low[lat_idx_low_start:lat_idx_low_end, lon_idx_low_start:lon_idx_low_end]
    zoom_high = grid_high_reconstructed[lat_idx_high_start:lat_idx_high_end, lon_idx_high_start:lon_idx_high_end]
    
    # Create zoomed comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Low-res zoom (show pixelation with nearest neighbor)
    im1 = axes[0].imshow(zoom_low, cmap='RdBu_r', aspect='auto', 
                         extent=[lon_zoom[0], lon_zoom[1], lat_zoom[0], lat_zoom[1]],
                         origin='upper', interpolation='nearest')  # nearest to show pixels
    axes[0].set_title(f'Input (Zoomed): L=25\n{zoom_low.shape[0]}×{zoom_low.shape[1]} pixels - Note the blockiness', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Longitude (degrees)', fontsize=12)
    axes[0].set_ylabel('Latitude (degrees)', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='-', linewidth=1, color='white')
    cbar1 = plt.colorbar(im1, ax=axes[0], label='Gravity (mGal)')
    
    # High-res zoom (show detail with nearest neighbor)
    im2 = axes[1].imshow(zoom_high, cmap='RdBu_r', aspect='auto',
                         extent=[lon_zoom[0], lon_zoom[1], lat_zoom[0], lat_zoom[1]],
                         origin='upper', interpolation='nearest')  # nearest to show pixels
    axes[1].set_title(f'Reconstructed (Zoomed): L=200\n{zoom_high.shape[0]}×{zoom_high.shape[1]} pixels - Much finer detail', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Longitude (degrees)', fontsize=12)
    axes[1].set_ylabel('Latitude (degrees)', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='-', linewidth=1, color='white')
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Gravity (mGal)')
    
    plt.suptitle(f'Zoomed Comparison: Region {lat_zoom[0]}°-{lat_zoom[1]}°N, {lon_zoom[0]}°-{lon_zoom[1]}°E\nPixel Detail Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    zoom_path = output_path.replace('.png', '_zoomed.png')
    plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved zoomed comparison to {zoom_path}")
    print(f"      The zoomed view clearly shows {zoom_high.shape[0]*zoom_high.shape[1] / (zoom_low.shape[0]*zoom_low.shape[1]):.1f}x more pixels in the reconstructed version")
    plt.close()


def main():
    """Main application function"""
    print("\n" + "="*80)
    print("MERCURY GRAVITY FIELD RECONSTRUCTION")
    print("Applying trained Moon model to Mercury data")
    print("="*80)
    
    # Paths
    model_path = 'checkpoints/moon_gravity_model_best.h5'
    norm_params_path = 'checkpoints/normalization_params.npz'
    mercury_data_path = 'data/processed/mercury_grav_L25.npz'
    output_dir = 'results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check files exist
    if not os.path.exists(model_path):
        print(f"\n✗ ERROR: Model not found at {model_path}")
        print("  Please train the model first using: python train_optimized.py")
        return
    
    if not os.path.exists(norm_params_path):
        print(f"\n✗ ERROR: Normalization params not found at {norm_params_path}")
        print("  Please train the model first using: python train_optimized.py")
        return
    
    if not os.path.exists(mercury_data_path):
        print(f"\n✗ ERROR: Mercury data not found at {mercury_data_path}")
        print("  Please run preprocessing first")
        return
    
    # Load model
    print(f"\n[1/5] Loading trained model...")
    print(f"      {model_path}")
    custom_objects = {
        'GravityReconstructionNetwork': GravityReconstructionNetwork,
        'DenseBlock': DenseBlock
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("      ✓ Model loaded")
    
    # Load normalization parameters
    print(f"\n[2/5] Loading normalization parameters...")
    norm_params = np.load(norm_params_path)
    low_mean = norm_params['low_mean']
    low_std = norm_params['low_std']
    high_mean = norm_params['high_mean']
    high_std = norm_params['high_std']
    print(f"      ✓ Parameters loaded")
    
    # Load Mercury data
    print(f"\n[3/5] Loading Mercury low-resolution data (L=25)...")
    grid_low_mercury = np.load(mercury_data_path)['grid']
    print(f"      Shape: {grid_low_mercury.shape}")
    print(f"      Range: [{np.min(grid_low_mercury):.2f}, {np.max(grid_low_mercury):.2f}] mGal")
    
    # Create patches and predict
    print(f"\n[4/5] Running reconstruction...")
    print(f"      Creating patches (stride=15)...")
    patches, positions, metadata = create_patches_from_grid(grid_low_mercury, patch_size=30, stride=15)
    print(f"      Extracted {len(patches)} patches")
    
    print(f"      Normalizing patches...")
    patches_norm = (patches - low_mean) / (low_std + 1e-8)
    
    print(f"      Predicting high-resolution patches...")
    predictions_norm = model.predict(patches_norm, batch_size=32, verbose=1)
    
    print(f"      Denormalizing predictions...")
    predictions = predictions_norm * high_std + high_mean
    
    print(f"      Reconstructing full grid...")
    reconstructed = reconstruct_from_patches(
        predictions, positions, metadata, patch_size=30
    )
    print(f"      ✓ Reconstruction complete")
    print(f"      Output shape: {reconstructed.shape}")
    print(f"      Output range: [{np.min(reconstructed):.2f}, {np.max(reconstructed):.2f}] mGal")
    
    # Save results
    print(f"\n[5/5] Saving results...")
    
    # Save reconstructed grid
    output_npz = f'{output_dir}/mercury_reconstructed_L200.npz'
    np.savez_compressed(output_npz, grid=reconstructed, lmax=200)
    print(f"      ✓ Saved data to {output_npz}")
    
    # Create visualization
    output_png = f'{output_dir}/mercury_reconstruction.png'
    visualize_mercury_reconstruction(grid_low_mercury, reconstructed, output_png)
    
    # Optional: Power spectrum analysis to check physical plausibility
    if METRICS_AVAILABLE:
        print(f"\n[BONUS] Running power spectrum analysis...")
        try:
            compare_power_spectra(
                grid_low_mercury, 
                reconstructed, 
                title_prefix="Mercury_"
            )
            print(f"      ✓ Power spectrum saved to results/mercury_power_spectrum.png")
        except Exception as e:
            print(f"      ⚠ Power spectrum analysis failed: {e}")
    
    print("\n" + "="*80)
    print("✓ RECONSTRUCTION COMPLETE!")
    print("="*80)
    print(f"\nResults saved:")
    print(f"  • Reconstructed grid: {output_npz}")
    print(f"  • Visualization:      {output_png}")
    if METRICS_AVAILABLE:
        print(f"  • Power spectrum:     results/mercury_power_spectrum.png")
    print("\nYou can now view the PNG file to see the Mercury reconstruction!")
    print("="*80)


if __name__ == "__main__":
    main()