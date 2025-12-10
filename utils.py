import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os


def visualize_gravity_field(gravity_field, title="Gravity Field",
                            save_path=None, cmap='RdBu_r', vmin=None, vmax=None):
    """
    Visualize a gravity field as a 2D map

    Args:
        gravity_field: 2D array of gravity data
        title: Plot title
        save_path: Path to save figure (optional)
        cmap: Colormap to use
        vmin, vmax: Value range for colormap
    """
    if len(gravity_field.shape) == 3 and gravity_field.shape[-1] == 1:
        gravity_field = gravity_field[:, :, 0]

    fig, ax = plt.subplots(figsize=(12, 8))

    if vmin is None or vmax is None:
        vmax_abs = np.abs(gravity_field).max()
        vmin, vmax = -vmax_abs, vmax_abs

    im = ax.imshow(gravity_field, cmap=cmap, origin='lower',
                   vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Gravity Anomaly (mGal)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def compare_gravity_fields(original, reconstructed, difference=None,
                          titles=None, save_path=None):
    """
    Compare original and reconstructed gravity fields side by side

    Args:
        original: Original gravity field
        reconstructed: Reconstructed gravity field
        difference: Difference field (optional, will compute if None)
        titles: List of titles for [original, reconstructed, difference]
        save_path: Path to save figure
    """
    if len(original.shape) == 3 and original.shape[-1] == 1:
        original = original[:, :, 0]
    if len(reconstructed.shape) == 3 and reconstructed.shape[-1] == 1:
        reconstructed = reconstructed[:, :, 0]

    if difference is None:
        difference = reconstructed - original

    if titles is None:
        titles = ['Original', 'Reconstructed', 'Difference']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax_abs = max(np.abs(original).max(), np.abs(reconstructed).max())
    vmin, vmax = -vmax_abs, vmax_abs

    im1 = axes[0].imshow(original, cmap='RdBu_r', origin='lower',
                        vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(reconstructed, cmap='RdBu_r', origin='lower',
                        vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    diff_max = np.abs(difference).max()
    im3 = axes[2].imshow(difference, cmap='RdBu_r', origin='lower',
                        vmin=-diff_max, vmax=diff_max, aspect='auto')
    axes[2].set_title(titles[2], fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('mGal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and metrics)

    Args:
        history: Keras training history object
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MAE Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if 'mae' in history.history:
        axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
        axes[0, 1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    if 'val_pearson' in history.history:
        axes[1, 0].plot(history.history['val_pearson'], label='Pearson r',
                       linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Pearson Correlation')
        axes[1, 0].set_title('Validation Pearson Correlation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    if 'val_ssim' in history.history:
        axes[1, 1].plot(history.history['val_ssim'], label='SSIM',
                       linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('Validation SSIM')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def plot_power_spectrum_comparison(original, reconstructed, degrees=None,
                                   title_prefix="", save_path=None):
    """
    Compare power spectra of original and reconstructed fields

    Args:
        original: Original gravity field
        reconstructed: Reconstructed gravity field
        degrees: Array of degree values (optional)
        title_prefix: Prefix for plot title
        save_path: Path to save figure
    """
    from model_eval import power_spectrum_analysis

    orig_spectrum = power_spectrum_analysis(original)
    recon_spectrum = power_spectrum_analysis(reconstructed)

    if degrees is None:
        degrees = np.arange(len(orig_spectrum))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.semilogy(degrees[:len(orig_spectrum)], orig_spectrum,
               label='Original', linewidth=2.5, alpha=0.8)
    ax.semilogy(degrees[:len(recon_spectrum)], recon_spectrum,
               label='Reconstructed', linewidth=2.5, alpha=0.8, linestyle='--')

    ax.set_xlabel('Spherical Harmonic Degree', fontsize=13)
    ax.set_ylabel('Power', fontsize=13)
    ax.set_title(f'{title_prefix}Power Spectrum Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def visualize_patches(patches, num_samples=16, title="Sample Patches", save_path=None):
    """
    Visualize a grid of sample patches

    Args:
        patches: Array of patches (num_patches, H, W, 1)
        num_samples: Number of patches to display
        title: Plot title
        save_path: Path to save figure
    """
    num_samples = min(num_samples, len(patches))
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        patch = patches[i]
        if len(patch.shape) == 3 and patch.shape[-1] == 1:
            patch = patch[:, :, 0]

        vmax_abs = np.abs(patch).max()
        axes[i].imshow(patch, cmap='RdBu_r', vmin=-vmax_abs, vmax=vmax_abs)
        axes[i].axis('off')
        axes[i].set_title(f'Patch {i+1}', fontsize=9)

    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    return fig


def save_results(original, reconstructed, metrics, output_dir='results'):
    """
    Save reconstruction results including visualizations and metrics

    Args:
        original: Original gravity field
        reconstructed: Reconstructed gravity field
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'original.npy'), original)
    np.save(os.path.join(output_dir, 'reconstructed.npy'), reconstructed)
    np.save(os.path.join(output_dir, 'difference.npy'), reconstructed - original)

    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("RECONSTRUCTION METRICS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    compare_gravity_fields(
        original, reconstructed,
        save_path=os.path.join(output_dir, 'comparison.png')
    )

    plot_power_spectrum_comparison(
        original, reconstructed,
        save_path=os.path.join(output_dir, 'power_spectrum.png')
    )


def calculate_degree_correlation(coeffs_true, coeffs_pred, max_degree=None):
    """
    Calculate correlation as a function of spherical harmonic degree

    Args:
        coeffs_true: Ground truth coefficients (C, S matrices)
        coeffs_pred: Predicted coefficients (C, S matrices)
        max_degree: Maximum degree to compute (None for all)

    Returns:
        Array of correlations by degree
    """
    C_true, S_true = coeffs_true
    C_pred, S_pred = coeffs_pred

    if max_degree is None:
        max_degree = min(C_true.shape[0], C_pred.shape[0]) - 1

    correlations = []

    for degree in range(max_degree + 1):
        c_true = C_true[degree, :degree+1]
        s_true = S_true[degree, 1:degree+1]
        c_pred = C_pred[degree, :degree+1]
        s_pred = S_pred[degree, 1:degree+1]

        true_coeffs = np.concatenate([c_true, s_true])
        pred_coeffs = np.concatenate([c_pred, s_pred])

        if len(true_coeffs) > 1 and np.std(true_coeffs) > 0 and np.std(pred_coeffs) > 0:
            corr = np.corrcoef(true_coeffs, pred_coeffs)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)

    return np.array(correlations)


def create_summary_report(metrics, history=None, output_path='summary_report.txt'):
    """
    Create a comprehensive summary report

    Args:
        metrics: Dictionary of evaluation metrics
        history: Training history (optional)
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MERCURY GRAVITY FIELD RECONSTRUCTION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("EVALUATION METRICS\n")
        f.write("-"*80 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:30s}: {value:.6f}\n")
            else:
                f.write(f"{key:30s}: {value}\n")
        f.write("\n")

        if history is not None:
            f.write("TRAINING SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total epochs trained: {len(history.history['loss'])}\n")
            f.write(f"Final training loss: {history.history['loss'][-1]:.6f}\n")
            f.write(f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n")
            f.write(f"Best validation loss: {min(history.history['val_loss']):.6f}\n")
            f.write("\n")

        f.write("="*80 + "\n")


if __name__ == "__main__":
    dummy_field = np.random.randn(180, 360) * 100
    visualize_gravity_field(dummy_field, title="Test Gravity Field")

    dummy_recon = dummy_field + np.random.randn(180, 360) * 10
    compare_gravity_fields(dummy_field, dummy_recon)

    dummy_patches = np.random.randn(16, 30, 30, 1) * 50
    visualize_patches(dummy_patches, title="Test Patches")
