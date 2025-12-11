"""Reconstruct Mercury's Southern Hemisphere"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import argparse

from moon_model import GravityReconstructionNetwork, DEMRefiningNetwork

CUSTOM_OBJECTS = {
    'GravityReconstructionNetwork': GravityReconstructionNetwork,
    'DEMRefiningNetwork': DEMRefiningNetwork
}

VISUALIZATION_CONFIG = {
    'cmap': 'RdBu_r',
    'vmin': -150,
    'vmax': 150,
    'upscale_factor': 4,
    'figsize': (15, 10),
    'dpi': 300
}


def load_model_safe(filepath):
    """Load model with custom objects"""
    try:
        return keras.models.load_model(filepath, custom_objects=CUSTOM_OBJECTS)
    except:
        return keras.models.load_model(filepath)

def sliding_window_reconstruction(model, grav_low, dem, patch_size=30, stride=15):
    """Runs the model over the full map using a sliding window"""
    h, w = grav_low.shape

    prediction_sum = np.zeros((h, w))
    overlap_count = np.zeros((h, w))

    print(f"Reconstructing map ({h}x{w}) with sliding window...")

    pad = patch_size // 2
    grav_padded = np.pad(grav_low, ((pad, pad), (pad, pad)), mode='reflect')
    dem_padded = np.pad(dem, ((pad, pad), (pad, pad)), mode='reflect')

    batch_grav = []
    batch_dem = []
    coords = []

    BATCH_SIZE = 64
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            g_patch = grav_padded[i:i+patch_size, j:j+patch_size]
            d_patch = dem_padded[i:i+patch_size, j:j+patch_size]

            batch_grav.append(g_patch)
            batch_dem.append(d_patch)
            coords.append((i, j))

            if len(batch_grav) >= BATCH_SIZE:
                bg = np.array(batch_grav)
                bd = np.array(batch_dem)

                bg = bg[..., np.newaxis]
                bd = bd[..., np.newaxis]

                preds = model.predict([bg, bd], verbose=0)
                preds = preds.squeeze()

                for idx, (r, c) in enumerate(coords):
                    p_h, p_w = preds[idx].shape

                    r_end = min(r + p_h, h)
                    c_end = min(c + p_w, w)

                    pr_end = r_end - r
                    pc_end = c_end - c

                    prediction_sum[r:r_end, c:c_end] += preds[idx, :pr_end, :pc_end]
                    overlap_count[r:r_end, c:c_end] += 1

                batch_grav = []
                batch_dem = []
                coords = []

    if batch_grav:
        bg = np.array(batch_grav)[..., np.newaxis]
        bd = np.array(batch_dem)[..., np.newaxis]
        preds = model.predict([bg, bd], verbose=0).squeeze()

        if len(batch_grav) == 1:
            preds = preds[np.newaxis, ...]

        for idx, (r, c) in enumerate(coords):
            p_h, p_w = preds[idx].shape
            r_end = min(r + p_h, h)
            c_end = min(c + p_w, w)
            pr_end = r_end - r
            pc_end = c_end - c
            prediction_sum[r:r_end, c:c_end] += preds[idx, :pr_end, :pc_end]
            overlap_count[r:r_end, c:c_end] += 1

    overlap_count[overlap_count == 0] = 1

    return prediction_sum / overlap_count


def prepare_visualization_data(grav_low, grav_truth, final_map, upscale_factor=4):
    """Upscale data for better visualization"""
    from scipy.ndimage import zoom

    grav_low_upscaled = zoom(grav_low, upscale_factor, order=3)
    grav_truth_upscaled = zoom(grav_truth, upscale_factor, order=3)
    final_map_upscaled = zoom(final_map, upscale_factor, order=3)

    return grav_low_upscaled, grav_truth_upscaled, final_map_upscaled


def create_reconstruction_plot(grav_low, grav_truth, final_map, config=VISUALIZATION_CONFIG):
    """Create 3-panel reconstruction comparison plot"""
    viz_args = {'cmap': config['cmap'], 'vmin': config['vmin'], 'vmax': config['vmax']}

    plt.figure(figsize=config['figsize'])

    plt.subplot(3, 1, 1)
    plt.title("Original Low-Res Input (L25) - Upscaled for Visualization")
    plt.imshow(grav_low, **viz_args, interpolation='bilinear')
    plt.colorbar(label="mGal")
    plt.ylabel('Latitude (degrees)')
    plt.yticks(ticks=[0, 50, 100, 150, 200], labels=['-90', '-45', '0', '45', '90'])

    plt.subplot(3, 1, 2)
    plt.title("Ground Truth (North Only) - Upscaled for Visualization")
    plt.imshow(grav_truth, **viz_args, interpolation='bilinear')
    plt.colorbar(label="mGal")
    plt.ylabel('Latitude (degrees)')
    plt.yticks(ticks=[0, 50, 100, 150, 200], labels=['-90', '-45', '0', '45', '90'])

    plt.subplot(3, 1, 3)
    plt.title("Final Hybrid Map: Truth (North) + AI Reconstruction (South)")
    plt.imshow(final_map, **viz_args, interpolation='bilinear')
    plt.colorbar(label="mGal")
    plt.ylabel('Latitude (degrees)')
    plt.yticks(ticks=[0, 50, 100, 150, 200], labels=['-90', '-45', '0', '45', '90'])

    plt.tight_layout()


def stitch_and_blend_hemispheres(grav_truth, reconstructed_map, blend_width=10):
    """Stitch north (truth) and south (predicted) with equator blending"""
    final_map = np.zeros_like(reconstructed_map)
    height = final_map.shape[0]
    equator = height // 2

    final_map[:equator, :] = grav_truth[:equator, :]
    final_map[equator:, :] = reconstructed_map[equator:, :]

    for i in range(blend_width):
        alpha = i / blend_width
        row = equator - blend_width // 2 + i
        if 0 <= row < height:
            final_map[row, :] = (1 - alpha) * grav_truth[row, :] + alpha * reconstructed_map[row, :]

    return final_map


def normalize_data(grav_low, dem, grav_truth):
    """Compute global normalization statistics"""
    g_mean, g_std = np.mean(grav_low), np.std(grav_low)
    t_mean, t_std = np.mean(grav_truth), np.std(grav_truth)
    d_mean, d_std = np.mean(dem), np.std(dem)

    print(f"Stats - Input Mean: {g_mean:.2f}, Std: {g_std:.2f}")
    print(f"Stats - Target Mean: {t_mean:.2f}, Std: {t_std:.2f}")

    grav_low_norm = (grav_low - g_mean) / (g_std + 1e-8)
    dem_norm = (dem - d_mean) / (d_std + 1e-8)

    return grav_low_norm, dem_norm, (t_mean, t_std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoints_mercury/mercury_model_best.h5')
    parser.add_argument('--grav_low', default='data/processed/mercury_grav_L25.npy')
    parser.add_argument('--grav_high_truth', default='data/processed/mercury_grav_L50.npy') # For the North
    parser.add_argument('--dem_high', default='data/processed/mercury_dem_720x360.npy')
    parser.add_argument('--output_dir', default='results_reconstruction')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    grav_low = np.load(args.grav_low)
    dem = np.load(args.dem_high)
    grav_truth = np.load(args.grav_high_truth)

    print(f"Shape Check - Gravity Low: {grav_low.shape}, Gravity Truth: {grav_truth.shape}, DEM: {dem.shape}")

    if grav_truth.shape != grav_low.shape:
        print(f"  Resizing Truth from {grav_truth.shape} to {grav_low.shape}...")
        from scipy.ndimage import zoom
        zoom_h = grav_low.shape[0] / grav_truth.shape[0]
        zoom_w = grav_low.shape[1] / grav_truth.shape[1]
        grav_truth = zoom(grav_truth, (zoom_h, zoom_w), order=1)

    if grav_low.shape != dem.shape:
        print(f"  Resizing DEM to match Gravity {grav_low.shape}...")
        from scipy.ndimage import zoom
        zoom_h = grav_low.shape[0] / dem.shape[0]
        zoom_w = grav_low.shape[1] / dem.shape[1]
        dem = zoom(dem, (zoom_h, zoom_w), order=1)

    print("Normalizing...")
    grav_low_norm, dem_norm, (t_mean, t_std) = normalize_data(grav_low, dem, grav_truth)

    print(f"Loading model: {args.model_path}")
    model = load_model_safe(args.model_path)

    print("Running reconstruction...")
    reconstructed_norm = sliding_window_reconstruction(model, grav_low_norm, dem_norm, stride=5)
    reconstructed_map = (reconstructed_norm * t_std) + t_mean

    from scipy.ndimage import gaussian_filter
    print("Applying smoothing...")
    reconstructed_map = gaussian_filter(reconstructed_map, sigma=0.1)

    print(f"Checking reconstruction...")
    print(f"  Reconstructed range: [{reconstructed_map.min():.2f}, {reconstructed_map.max():.2f}]")

    if abs(reconstructed_map.min()) > 200 or abs(reconstructed_map.max()) > 200:
        print(f"  WARNING: Extreme values detected! Clamping to safe range.")
        reconstructed_map = np.clip(reconstructed_map, -150, 150)

    print("Stitching North (Truth) and South (Predicted)...")
    final_map = stitch_and_blend_hemispheres(grav_truth, reconstructed_map)

    print("Saving results...")
    grav_low_up, grav_truth_up, final_map_up = prepare_visualization_data(
        grav_low, grav_truth, final_map, VISUALIZATION_CONFIG['upscale_factor']
    )

    create_reconstruction_plot(grav_low_up, grav_truth_up, final_map_up)

    viz_path = f"{args.output_dir}/mercury_reconstruction_comparison.pdf"
    plt.savefig(viz_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    print(f"\nSaved visualization to {viz_path}")

    np.save(f"{args.output_dir}/mercury_final_hybrid_map.npy", final_map)
    print(f"Saved hybrid map to {args.output_dir}/mercury_final_hybrid_map.npy")

if __name__ == "__main__":
    main()