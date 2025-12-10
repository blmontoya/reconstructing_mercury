"""
Evaluation Script for DEM-Enhanced Gravity Reconstruction
Compare gravity-only vs. gravity+DEM models
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
from skimage.metrics import structural_similarity as ssim


def load_model_and_data(
    model_path,
    gravity_low_path,
    gravity_high_path,
    dem_high_path=None,
    patch_size=30
):
    """
    Load trained model and test data

    Args:
        model_path: Path to saved model
        gravity_low_path: Path to low-res gravity test data
        gravity_high_path: Path to high-res gravity ground truth
        dem_high_path: Path to high-res DEM (optional, for full model)
        patch_size: Patch size for evaluation

    Returns:
        model, test_data, ground_truth, (dem_data if applicable)
    """
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)

    print(f"Loading gravity data...")
    gravity_low = np.load(gravity_low_path)
    gravity_high = np.load(gravity_high_path)

    # Resize if needed
    if gravity_low.shape != gravity_high.shape:
        zoom_factors = (gravity_high.shape[0] / gravity_low.shape[0],
                       gravity_high.shape[1] / gravity_low.shape[1])
        gravity_low = zoom(gravity_low, zoom_factors, order=1)

    dem_high = None
    if dem_high_path:
        print(f"Loading DEM data...")
        dem_high = np.load(dem_high_path)
        if dem_high.shape != gravity_high.shape:
            zoom_factors = (gravity_high.shape[0] / dem_high.shape[0],
                           gravity_high.shape[1] / dem_high.shape[1])
            dem_high = zoom(dem_high, zoom_factors, order=1)

    print(f"  Gravity low shape: {gravity_low.shape}")
    print(f"  Gravity high shape: {gravity_high.shape}")
    if dem_high is not None:
        print(f"  DEM high shape: {dem_high.shape}")

    return model, gravity_low, gravity_high, dem_high


def predict_full_field(model, gravity_low, dem_high=None, patch_size=30, stride=15):
    """
    Predict full gravity field using sliding window

    Args:
        model: Trained model
        gravity_low: Low-resolution gravity input
        dem_high: High-resolution DEM (optional)
        patch_size: Patch size
        stride: Stride for sliding window

    Returns:
        Reconstructed high-resolution gravity field
    """
    print(f"\nPredicting full field with sliding window...")
    print(f"  Patch size: {patch_size}x{patch_size}, Stride: {stride}")

    h, w = gravity_low.shape
    output = np.zeros_like(gravity_low)
    counts = np.zeros_like(gravity_low)

    has_dem = dem_high is not None

    total_patches = ((h - patch_size) // stride + 1) * ((w - patch_size) // stride + 1)
    processed = 0

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Extract patches
            grav_patch = gravity_low[i:i+patch_size, j:j+patch_size]
            grav_patch = grav_patch[np.newaxis, ..., np.newaxis]

            if has_dem:
                dem_patch = dem_high[i:i+patch_size, j:j+patch_size]
                dem_patch = dem_patch[np.newaxis, ..., np.newaxis]
                pred = model.predict([grav_patch, dem_patch], verbose=0)
            else:
                pred = model.predict(grav_patch, verbose=0)

            # Accumulate prediction
            output[i:i+patch_size, j:j+patch_size] += pred[0, :, :, 0]
            counts[i:i+patch_size, j:j+patch_size] += 1

            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{total_patches} patches...")

    # Average overlapping predictions
    output = output / (counts + 1e-8)

    print(f"  Prediction complete!")

    return output


def calculate_metrics(prediction, ground_truth):
    """
    Calculate comprehensive evaluation metrics

    Args:
        prediction: Predicted gravity field
        ground_truth: Ground truth gravity field

    Returns:
        Dictionary of metrics
    """
    print(f"\nCalculating evaluation metrics...")

    # Flatten arrays for metric calculations
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()

    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(gt_flat))
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]

    # 1. Pearson Correlation
    pearson = np.corrcoef(pred_flat, gt_flat)[0, 1]

    # 2. SSIM
    data_range = ground_truth.max() - ground_truth.min()
    ssim_value = ssim(ground_truth, prediction, data_range=data_range)

    # 3. RMSE
    rmse = np.sqrt(np.mean((pred_flat - gt_flat) ** 2))

    # 4. MAE
    mae = np.mean(np.abs(pred_flat - gt_flat))

    # 5. Relative Error
    relative_error = np.mean(np.abs((pred_flat - gt_flat) / (np.abs(gt_flat) + 1e-8))) * 100

    metrics = {
        'Pearson Correlation': pearson,
        'SSIM': ssim_value,
        'RMSE (mGal)': rmse,
        'MAE (mGal)': mae,
        'Relative Error (%)': relative_error
    }

    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key:25s}: {value:.6f}")
    print("="*60)

    return metrics


def visualize_results(gravity_low, prediction, ground_truth, dem_high=None, save_dir='results_dem'):
    """
    Create comprehensive visualizations

    Args:
        gravity_low: Low-resolution input
        prediction: Model prediction
        ground_truth: Ground truth
        dem_high: High-resolution DEM (optional)
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    # Figure 1: Main comparison
    if dem_high is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

    vmax = max(np.abs(prediction).max(), np.abs(ground_truth).max())
    vmin = -vmax

    # Low-res input
    im0 = axes[0].imshow(gravity_low, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('Low-Resolution Input', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im0, ax=axes[0], label='mGal')

    # Ground truth
    im1 = axes[1].imshow(ground_truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title('Ground Truth (High-Res)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[1], label='mGal')

    # Prediction
    im2 = axes[2].imshow(prediction, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title('Model Prediction', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[2], label='mGal')

    # Difference
    difference = prediction - ground_truth
    diff_max = np.abs(difference).max()
    im3 = axes[3].imshow(difference, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, aspect='auto')
    axes[3].set_title(f'Difference (Pred - GT)\nRMSE: {np.sqrt(np.mean(difference**2)):.4f} mGal',
                     fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Longitude')
    axes[3].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[3], label='mGal')

    if dem_high is not None:
        # DEM
        im4 = axes[4].imshow(dem_high, cmap='terrain', aspect='auto')
        axes[4].set_title('High-Resolution DEM', fontsize=14, fontweight='bold')
        axes[4].set_xlabel('Longitude')
        axes[4].set_ylabel('Latitude')
        plt.colorbar(im4, ax=axes[4], label='Elevation (normalized)')

        # Scatter plot: Prediction vs Ground Truth
        sample_mask = np.random.choice(prediction.size, size=min(10000, prediction.size), replace=False)
        pred_sample = prediction.flatten()[sample_mask]
        gt_sample = ground_truth.flatten()[sample_mask]

        axes[5].scatter(gt_sample, pred_sample, alpha=0.3, s=1)
        axes[5].plot([gt_sample.min(), gt_sample.max()],
                    [gt_sample.min(), gt_sample.max()],
                    'r--', linewidth=2, label='Perfect fit')
        axes[5].set_xlabel('Ground Truth (mGal)', fontsize=12)
        axes[5].set_ylabel('Prediction (mGal)', fontsize=12)
        axes[5].set_title('Prediction vs Ground Truth', fontsize=14, fontweight='bold')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/full_comparison.png', dpi=200, bbox_inches='tight')
    print(f"\n  Saved comparison to {save_dir}/full_comparison.png")
    plt.close()


def compare_gravity_only_vs_dem(
    gravity_only_model_path,
    dem_model_path,
    gravity_low_path,
    gravity_high_path,
    dem_high_path
):
    """
    Compare gravity-only model vs. gravity+DEM model

    Args:
        gravity_only_model_path: Path to gravity-only trained model
        dem_model_path: Path to gravity+DEM trained model
        gravity_low_path: Path to test low-res gravity
        gravity_high_path: Path to test high-res gravity
        dem_high_path: Path to test high-res DEM
    """
    print("\n" + "="*80)
    print("COMPARING GRAVITY-ONLY vs GRAVITY+DEM MODELS")
    print("="*80)

    # Load gravity-only model
    print("\n--- GRAVITY-ONLY MODEL ---")
    model_gravity, grav_low, grav_high, _ = load_model_and_data(
        gravity_only_model_path, gravity_low_path, gravity_high_path
    )

    pred_gravity_only = predict_full_field(model_gravity, grav_low, dem_high=None)
    metrics_gravity = calculate_metrics(pred_gravity_only, grav_high)

    # Load DEM model
    print("\n--- GRAVITY+DEM MODEL ---")
    model_dem, _, _, dem_high = load_model_and_data(
        dem_model_path, gravity_low_path, gravity_high_path, dem_high_path
    )

    pred_dem = predict_full_field(model_dem, grav_low, dem_high=dem_high)
    metrics_dem = calculate_metrics(pred_dem, grav_high)

    # Comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Metric':<25s} {'Gravity-Only':>15s} {'Gravity+DEM':>15s} {'Improvement':>15s}")
    print("-"*80)

    for key in metrics_gravity.keys():
        val_g = metrics_gravity[key]
        val_d = metrics_dem[key]

        if 'Correlation' in key or 'SSIM' in key:
            improvement = ((val_d - val_g) / val_g) * 100
        else:  # Lower is better for RMSE, MAE
            improvement = ((val_g - val_d) / val_g) * 100

        print(f"{key:<25s} {val_g:>15.6f} {val_d:>15.6f} {improvement:>14.2f}%")

    print("="*80)

    # Visualize both
    visualize_results(grav_low, pred_gravity_only, grav_high, dem_high=None,
                     save_dir='results_gravity_only')
    visualize_results(grav_low, pred_dem, grav_high, dem_high=dem_high,
                     save_dir='results_gravity_dem')

    return metrics_gravity, metrics_dem


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate DEM-enhanced gravity reconstruction')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--gravity_low', type=str, required=True,
                       help='Path to low-res gravity test data')
    parser.add_argument('--gravity_high', type=str, required=True,
                       help='Path to high-res gravity ground truth')
    parser.add_argument('--dem_high', type=str, default=None,
                       help='Path to high-res DEM (optional)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare gravity-only vs gravity+DEM models')
    parser.add_argument('--gravity_only_model', type=str,
                       help='Path to gravity-only model (for comparison)')

    args = parser.parse_args()

    if args.compare and args.gravity_only_model:
        # Comparison mode
        compare_gravity_only_vs_dem(
            args.gravity_only_model,
            args.model_path,
            args.gravity_low,
            args.gravity_high,
            args.dem_high
        )
    else:
        # Single model evaluation
        model, grav_low, grav_high, dem_high = load_model_and_data(
            args.model_path,
            args.gravity_low,
            args.gravity_high,
            args.dem_high
        )

        prediction = predict_full_field(model, grav_low, dem_high=dem_high)
        metrics = calculate_metrics(prediction, grav_high)

        save_dir = 'results_dem' if dem_high is not None else 'results_gravity'
        visualize_results(grav_low, prediction, grav_high, dem_high, save_dir=save_dir)

        print(f"\nResults saved to {save_dir}/")
