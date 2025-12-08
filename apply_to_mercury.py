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
from model import GravityReconstructionNetwork, DenseBlock

# Import evaluation metrics (optional - for validation if you have ground truth)
try:
    from metrics import pearson_correlation, calculate_ssim, calculate_rmse, compare_power_spectra
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: metrics.py not found. Skipping evaluation metrics.")
    METRICS_AVAILABLE = False


def create_patches_from_grid(grid, patch_size=30, stride=15):
    """Extract overlapping patches from grid for prediction"""
    h, w = grid.shape
    patches = []
    positions = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = grid[i:i+patch_size, j:j+patch_size]
            
            if not np.isnan(patch).any():
                patches.append(patch)
                positions.append((i, j))
    
    patches = np.array(patches, dtype=np.float32)[..., np.newaxis]
    return patches, positions


def reconstruct_from_patches(patches, positions, original_shape, patch_size=30):
    """Reconstruct full grid from overlapping patches using averaging"""
    h, w = original_shape
    patches = patches.squeeze()
    
    reconstructed = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    for patch, (i, j) in zip(patches, positions):
        reconstructed[i:i+patch_size, j:j+patch_size] += patch
        counts[i:i+patch_size, j:j+patch_size] += 1
    
    counts[counts == 0] = 1
    reconstructed /= counts
    
    return reconstructed


def visualize_mercury_reconstruction(grid_low, grid_high_reconstructed, output_path):
    """Create before/after visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Input (Low Resolution)
    im1 = axes[0].imshow(grid_low, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Input: Mercury L=25\n(Low Resolution)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Longitude', fontsize=12)
    axes[0].set_ylabel('Latitude', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=axes[0], label='Gravity (mGal)', fraction=0.046)
    cbar1.ax.tick_params(labelsize=10)
    
    # Output (Reconstructed High Resolution)
    im2 = axes[1].imshow(grid_high_reconstructed, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('Reconstructed: Mercury L=200\n(High Resolution)', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Longitude', fontsize=12)
    axes[1].set_ylabel('Latitude', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Gravity (mGal)', fraction=0.046)
    cbar2.ax.tick_params(labelsize=10)
    
    plt.suptitle('Mercury Gravity Field Reconstruction\nTrained on Moon (GRAIL) → Applied to Mercury (MESSENGER)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
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
    patches, positions = create_patches_from_grid(grid_low_mercury, patch_size=30, stride=15)
    print(f"      Extracted {len(patches)} patches")
    
    print(f"      Normalizing patches...")
    patches_norm = (patches - low_mean) / (low_std + 1e-8)
    
    print(f"      Predicting high-resolution patches...")
    predictions_norm = model.predict(patches_norm, batch_size=32, verbose=1)
    
    print(f"      Denormalizing predictions...")
    predictions = predictions_norm * high_std + high_mean
    
    print(f"      Reconstructing full grid...")
    reconstructed = reconstruct_from_patches(
        predictions, positions, grid_low_mercury.shape, patch_size=30
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