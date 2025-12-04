import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim


def pearson_correlation(y_true, y_pred):
    """calculate pearson correlation coefficient"""
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    corr, _ = pearsonr(y_true_flat, y_pred_flat)
    return corr


def calculate_ssim(y_true, y_pred, data_range=None):
    """calculate structural similarity index"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if data_range is None:
        data_range = max(y_true.max() - y_true.min(),
                        y_pred.max() - y_pred.min())

    if len(y_true.shape) == 4:
        ssim_scores = []
        for i in range(y_true.shape[0]):
            img_true = y_true[i, :, :, 0] if y_true.shape[-1] == 1 else y_true[i]
            img_pred = y_pred[i, :, :, 0] if y_pred.shape[-1] == 1 else y_pred[i]
            score = ssim(img_true, img_pred, data_range=data_range)
            ssim_scores.append(score)
        return np.mean(ssim_scores)
    elif len(y_true.shape) == 3:
        img_true = y_true[:, :, 0] if y_true.shape[-1] == 1 else y_true
        img_pred = y_pred[:, :, 0] if y_pred.shape[-1] == 1 else y_pred
        return ssim(img_true, img_pred, data_range=data_range)
    else:
        return ssim(y_true, y_pred, data_range=data_range)


def calculate_rmse(y_true, y_pred):
    """calculate root mean square error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def evaluate_model(model, test_data, test_labels, batch_size=32, verbose=True):
    """comprehensive evaluation of the model using all three metrics"""
    if verbose:
        print("generating predictions...")
    predictions = model.predict(test_data, batch_size=batch_size, verbose=1 if verbose else 0)

    if verbose:
        print("calculating metrics...")

    pearson = pearson_correlation(test_labels, predictions)
    ssim_score = calculate_ssim(test_labels, predictions)
    rmse = calculate_rmse(test_labels, predictions)

    results = {
        'pearson_correlation': pearson,
        'ssim': ssim_score,
        'rmse': rmse,
        'num_samples': len(test_labels)
    }

    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Number of test samples: {results['num_samples']}")
        print(f"Pearson Correlation (r): {pearson:.6f}")
        print(f"SSIM: {ssim_score:.6f}")
        print(f"RMSE: {rmse:.6f} mGal")
        print("="*60 + "\n")

    return results


def evaluate_batch(y_true_batch, y_pred_batch):
    """quick evaluation of a single batch"""
    return {
        'pearson': pearson_correlation(y_true_batch, y_pred_batch),
        'ssim': calculate_ssim(y_true_batch, y_pred_batch),
        'rmse': calculate_rmse(y_true_batch, y_pred_batch)
    }


class MetricsCallback(tf.keras.callbacks.Callback):
    """custom callback to track pearson, ssim, and rmse during training"""
    def __init__(self, validation_data, log_frequency=1):
        super(MetricsCallback, self).__init__()
        self.validation_data = validation_data
        self.log_frequency = log_frequency
        self.history = {
            'pearson': [],
            'ssim': [],
            'rmse': []
        }

    def on_epoch_end(self, epoch, logs=None):
        """calculate and log metrics at end of epoch"""
        if (epoch + 1) % self.log_frequency != 0:
            return

        val_inputs, val_labels = self.validation_data
        val_predictions = self.model.predict(val_inputs, verbose=0)

        pearson = pearson_correlation(val_labels, val_predictions)
        ssim_score = calculate_ssim(val_labels, val_predictions)
        rmse = calculate_rmse(val_labels, val_predictions)

        self.history['pearson'].append(pearson)
        self.history['ssim'].append(ssim_score)
        self.history['rmse'].append(rmse)

        logs['val_pearson'] = pearson
        logs['val_ssim'] = ssim_score
        logs['val_rmse'] = rmse

        print(f"\nvalidation metrics - pearson: {pearson:.4f}, ssim: {ssim_score:.4f}, rmse: {rmse:.4f}")


def power_spectrum_analysis(gravity_field):
    """compute power spectrum of gravity field for physical plausibility check"""
    if len(gravity_field.shape) == 3 and gravity_field.shape[-1] == 1:
        gravity_field = gravity_field[:, :, 0]

    fft_2d = np.fft.fft2(gravity_field)
    power_2d = np.abs(fft_2d) ** 2

    h, w = power_2d.shape
    center_y, center_x = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)

    r_max = min(center_y, center_x)
    power_spectrum = np.zeros(r_max)

    for i in range(r_max):
        mask = (r == i)
        if np.sum(mask) > 0:
            power_spectrum[i] = np.mean(power_2d[mask])

    return power_spectrum


def compare_power_spectra(original, reconstructed, title_prefix=""):
    """compare power spectra of original and reconstructed fields"""
    orig_spectrum = power_spectrum_analysis(original)
    recon_spectrum = power_spectrum_analysis(reconstructed)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.semilogy(orig_spectrum, label='Original', linewidth=2)
        plt.semilogy(recon_spectrum, label='Reconstructed', linewidth=2)
        plt.xlabel('Degree')
        plt.ylabel('Power')
        plt.title(f'{title_prefix}Power Spectrum Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{title_prefix.lower().replace(" ", "_")}power_spectrum.png', dpi=300)
        print(f"Power spectrum plot saved to {title_prefix.lower().replace(' ', '_')}power_spectrum.png")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")

    return orig_spectrum, recon_spectrum


if __name__ == "__main__":
    print("testing evaluation metrics...")

    y_true = np.random.randn(10, 30, 30, 1)
    y_pred = y_true + np.random.randn(10, 30, 30, 1) * 0.1

    print("\ntesting pearson correlation...")
    pearson = pearson_correlation(y_true, y_pred)
    print(f"pearson correlation: {pearson:.6f}")

    print("\ntesting ssim...")
    ssim_score = calculate_ssim(y_true, y_pred)
    print(f"ssim: {ssim_score:.6f}")

    print("\ntesting rmse...")
    rmse = calculate_rmse(y_true, y_pred)
    print(f"rmse: {rmse:.6f}")

    print("\ntesting batch evaluation...")
    batch_results = evaluate_batch(y_true, y_pred)
    print(f"batch results: {batch_results}")

    print("\ntesting power spectrum analysis...")
    spectrum = power_spectrum_analysis(y_true[0])
    print(f"power spectrum shape: {spectrum.shape}")

    print("\nall metrics tested successfully!")
