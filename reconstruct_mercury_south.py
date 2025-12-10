"""
Step 7: Reconstruct Mercury's Southern Hemisphere
- Loads fine-tuned model
- Generates high-res gravity for the South (Lat -90 to 0)
- Stitches it with the known North (Lat 0 to 90)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import argparse

# Import custom layers just like in training
from model import GravityReconstructionNetwork, DEMRefiningNetwork

def load_model_safe(filepath):
    """Load model with custom objects"""
    custom_objects = {
        'GravityReconstructionNetwork': GravityReconstructionNetwork,
        'DEMRefiningNetwork': DEMRefiningNetwork
    }
    try:
        return keras.models.load_model(filepath, custom_objects=custom_objects)
    except:
        return keras.models.load_model(filepath)

def sliding_window_reconstruction(model, grav_low, dem, patch_size=30, stride=15):
    """
    Runs the model over the full map using a sliding window.
    Averages predictions in overlapping regions for smoothness.
    """
    h, w = grav_low.shape
    
    # Buffers to hold the sum of predictions and the count of overlaps
    prediction_sum = np.zeros((h, w))
    overlap_count = np.zeros((h, w))
    
    print(f"Reconstructing map ({h}x{w}) with sliding window...")
    
    # Pad image to handle edges (simplified: reflection padding)
    pad = patch_size // 2
    grav_padded = np.pad(grav_low, ((pad, pad), (pad, pad)), mode='reflect')
    dem_padded = np.pad(dem, ((pad, pad), (pad, pad)), mode='reflect')
    
    # Sliding window loop
    # Note: This loops over pixels. For speed, we batch these in reality, 
    # but this loop is clearer for logic demonstration.
    
    batch_grav = []
    batch_dem = []
    coords = []
    
    BATCH_SIZE = 64
    
    # Grid generation
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            # Extract patches from padded array
            g_patch = grav_padded[i:i+patch_size, j:j+patch_size]
            d_patch = dem_padded[i:i+patch_size, j:j+patch_size]
            
            batch_grav.append(g_patch)
            batch_dem.append(d_patch)
            coords.append((i, j))
            
            # When batch is full, predict
            if len(batch_grav) >= BATCH_SIZE:
                # Convert to numpy
                bg = np.array(batch_grav)
                bd = np.array(batch_dem)
                
                # Reshape for model (Batch, H, W, 1)
                bg = bg[..., np.newaxis]
                bd = bd[..., np.newaxis]
                
                # Predict
                preds = model.predict([bg, bd], verbose=0)
                preds = preds.squeeze() # Remove extra dims
                
                # Place predictions back into the map
                for idx, (r, c) in enumerate(coords):
                    # Add prediction to sum buffer
                    # Note: Handle case where model output size != input size
                    # Assuming output is same size as input here
                    p_h, p_w = preds[idx].shape
                    
                    # Crop if going out of bounds (bottom/right edges)
                    r_end = min(r + p_h, h)
                    c_end = min(c + p_w, w)
                    
                    # How much to take from prediction
                    pr_end = r_end - r
                    pc_end = c_end - c
                    
                    prediction_sum[r:r_end, c:c_end] += preds[idx, :pr_end, :pc_end]
                    overlap_count[r:r_end, c:c_end] += 1
                
                # Reset batch
                batch_grav = []
                batch_dem = []
                coords = []

    # Process remaining batch
    if batch_grav:
        bg = np.array(batch_grav)[..., np.newaxis]
        bd = np.array(batch_dem)[..., np.newaxis]
        preds = model.predict([bg, bd], verbose=0).squeeze()
        
        if len(batch_grav) == 1: # Handle single item batch edge case
            preds = preds[np.newaxis, ...]

        for idx, (r, c) in enumerate(coords):
            p_h, p_w = preds[idx].shape
            r_end = min(r + p_h, h)
            c_end = min(c + p_w, w)
            pr_end = r_end - r
            pc_end = c_end - c
            prediction_sum[r:r_end, c:c_end] += preds[idx, :pr_end, :pc_end]
            overlap_count[r:r_end, c:c_end] += 1

    # Avoid division by zero
    overlap_count[overlap_count == 0] = 1
    
    return prediction_sum / overlap_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='checkpoints_mercury/mercury_model_best.h5')
    parser.add_argument('--grav_low', default='data/processed/mercury_grav_L25.npy')
    parser.add_argument('--grav_high_truth', default='data/processed/mercury_grav_L50.npy') # For the North
    parser.add_argument('--dem_high', default='data/processed/mercury_dem_720x360.npy')
    parser.add_argument('--output_dir', default='results_reconstruction')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    grav_low = np.load(args.grav_low)
    dem = np.load(args.dem_high)
    grav_truth = np.load(args.grav_high_truth) # We only trust the North half of this

    # --- INSERT THIS SAFETY BLOCK BEFORE STEP 2 ---
    print(f"Shape Check - Gravity Low: {grav_low.shape}, Gravity Truth: {grav_truth.shape}, DEM: {dem.shape}")
    
    # CRITICAL: Resize truth to match low-res input
    if grav_truth.shape != grav_low.shape:
        print(f"  ! Truth/Low mismatch. Resizing Truth from {grav_truth.shape} to {grav_low.shape}...")
        from scipy.ndimage import zoom
        zoom_h = grav_low.shape[0] / grav_truth.shape[0]
        zoom_w = grav_low.shape[1] / grav_truth.shape[1]
        grav_truth = zoom(grav_truth, (zoom_h, zoom_w), order=1)
        print(f"  ! New Truth shape: {grav_truth.shape}")
    
    if grav_low.shape != dem.shape:
        print(f"  ! DEM mismatch. Resizing DEM to match Gravity {grav_low.shape}...")
        from scipy.ndimage import zoom
        zoom_h = grav_low.shape[0] / dem.shape[0]
        zoom_w = grav_low.shape[1] / dem.shape[1]
        dem = zoom(dem, (zoom_h, zoom_w), order=1)
        print(f"  ! New DEM shape: {dem.shape}")
    # ----------------------------------------------

    # DEFINE MIDPOINT HERE (before using it)
    midpoint = grav_low.shape[0] // 2

    print("\n" + "="*60)
    print("HEMISPHERE VERIFICATION CHECK")
    print("="*60)

    print("\n" + "="*60)
    print("HEMISPHERE VERIFICATION CHECK")
    print("="*60)

    # Check pole values
    print("\n1. Pole Gravity Values:")
    print(f"   First row (should be South Pole -90°): {grav_low[0, :].mean():.2f} mGal")
    print(f"   Last row (should be North Pole +90°): {grav_low[-1, :].mean():.2f} mGal")
    print(f"   Middle row (Equator 0°): {grav_low[midpoint, :].mean():.2f} mGal")

    # Check which pole has lower gravity (physics check)
    first_10_rows = grav_low[:10, :].mean()
    last_10_rows = grav_low[-10:, :].mean()
    middle_10_rows = grav_low[midpoint-5:midpoint+5, :].mean()

    print("\n2. Regional Averages:")
    print(f"   Top 10 rows mean: {first_10_rows:.2f} mGal")
    print(f"   Middle 10 rows mean: {middle_10_rows:.2f} mGal")
    print(f"   Bottom 10 rows mean: {last_10_rows:.2f} mGal")

    print("\n3. Physics Check (poles should have LOWER gravity than equator):")
    if first_10_rows < middle_10_rows and last_10_rows < middle_10_rows:
        print("   ✓ PASS: Both poles have lower gravity than equator")
    else:
        print("   ✗ FAIL: Unexpected gravity distribution!")

    print("\n4. Hemisphere Assignment:")
    print(f"   Rows 0 to {midpoint-1}: Should be SOUTH (-90° to 0°)")
    print(f"   Rows {midpoint} to {grav_low.shape[0]-1}: Should be NORTH (0° to +90°)")

    print("\n5. Data Coverage Check:")
    north_std = grav_low[midpoint:, :].std()
    south_std = grav_low[:midpoint, :].std()
    print(f"   Northern hemisphere std dev: {north_std:.2f}")
    print(f"   Southern hemisphere std dev: {south_std:.2f}")
    print(f"   → MESSENGER data (detailed) should be in: {'NORTH' if north_std > south_std else 'SOUTH'}")

    print("="*60 + "\n")

    print("\n" + "="*60)
    print("DETAILED DATA INSPECTION")
    print("="*60)

    # Visual pattern check
    print("\n1. Visual Pattern Analysis:")
    print("   Checking if each hemisphere has smooth/detailed patterns...")

    # Calculate variation (how much detail/features exist)
    def calculate_variation(data):
        """Higher variation = more features/detail"""
        # Use Laplacian to measure local changes (edges/features)
        from scipy.ndimage import laplace
        lap = laplace(data)
        return np.abs(lap).mean()

    top_variation = calculate_variation(grav_low[:midpoint, :])
    bottom_variation = calculate_variation(grav_low[midpoint:, :])

    print(f"   Top half variation (edge detection): {top_variation:.4f}")
    print(f"   Bottom half variation (edge detection): {bottom_variation:.4f}")
    print(f"   → More variation = more features = MESSENGER data")
    print(f"   → MESSENGER data appears to be in: {'TOP' if top_variation > bottom_variation else 'BOTTOM'} half")

    # Value range check
    print("\n2. Value Range Check:")
    top_min, top_max = grav_low[:midpoint, :].min(), grav_low[:midpoint, :].max()
    bottom_min, bottom_max = grav_low[midpoint:, :].min(), grav_low[midpoint:, :].max()

    print(f"   Top half range: [{top_min:.2f}, {top_max:.2f}] mGal (span: {top_max - top_min:.2f})")
    print(f"   Bottom half range: [{bottom_min:.2f}, {bottom_max:.2f}] mGal (span: {bottom_max - bottom_min:.2f})")
    print(f"   → Larger range typically = more detail")

    # Histogram comparison
    print("\n3. Histogram Shapes:")
    print("   Creating histograms to compare distributions...")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(grav_low[:midpoint, :].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Top Half (rows 0-{midpoint-1})\nStd: {south_std:.2f}')
    axes[0].set_xlabel('Gravity (mGal)')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(grav_low[midpoint:, :].flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_title(f'Bottom Half (rows {midpoint}-{grav_low.shape[0]-1})\nStd: {north_std:.2f}')
    axes[1].set_xlabel('Gravity (mGal)')

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/hemisphere_comparison_histograms.png", dpi=150)
    print(f"   Saved histogram comparison to {args.output_dir}/hemisphere_comparison_histograms.png")

    # Visual map comparison
    print("\n4. Saving visual comparison of both halves...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].imshow(grav_low[:midpoint, :], cmap='RdBu_r', vmin=-100, vmax=100)
    axes[0].set_title(f'Top Half (rows 0-{midpoint-1}) - Std: {south_std:.2f}')
    axes[0].set_ylabel('Row')

    axes[1].imshow(grav_low[midpoint:, :], cmap='RdBu_r', vmin=-100, vmax=100)
    axes[1].set_title(f'Bottom Half (rows {midpoint}-end) - Std: {north_std:.2f}')
    axes[1].set_ylabel('Row')
    axes[1].set_xlabel('Column')

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/hemisphere_visual_comparison.png", dpi=200)
    print(f"   Saved visual comparison to {args.output_dir}/hemisphere_visual_comparison.png")

    print("\n5. CONCLUSION:")
    if south_std > north_std and top_variation > bottom_variation:
        print("   ✓ TOP half has more detail (higher std AND variation)")
        print("   ✓ This is where MESSENGER data is")
        print("   → Train on: rows [:midpoint]")
        print("   → Reconstruct: rows [midpoint:]")
    else:
        print("   ✓ BOTTOM half has more detail")
        print("   ✓ This is where MESSENGER data is")
        print("   → Train on: rows [midpoint:]")
        print("   → Reconstruct: rows [:midpoint]")

    print("="*60 + "\n")
    
    
    # 2. Normalize (FIXED: Use Northern Hemisphere Stats)
    print("Normalizing...")

    # Use statistics from the ENTIRE planet, not just one hemisphere
    g_mean = np.mean(grav_low)  # Changed from [:midpoint]
    g_std = np.std(grav_low)

    t_mean = np.mean(grav_truth)  # Changed from [:midpoint]
    t_std = np.std(grav_truth)
        
    print(f"Stats - Input Mean: {g_mean:.2f}, Std: {g_std:.2f} (GLOBAL)")
    print(f"Stats - Target Mean: {t_mean:.2f}, Std: {t_std:.2f} (GLOBAL)")

    # Apply these global stats to the WHOLE map
    grav_low_norm = (grav_low - g_mean) / (g_std + 1e-8)
    dem_norm = (dem - np.mean(dem)) / (np.std(dem) + 1e-8)

    # 3. Load Model
    print(f"Loading model: {args.model_path}")
    model = load_model_safe(args.model_path)

    # 4. Run Inference on GLOBAL Map (or just South)
    # We run globally to check consistency, then stitch.
    print("Running reconstruction (this may take a moment)...")
    reconstructed_norm = sliding_window_reconstruction(model, grav_low_norm, dem_norm, stride=5)  # Reduced from 15 to 10 for smoother blending
    
    # Denormalize
    reconstructed_map = (reconstructed_norm * t_std) + t_mean
    
    # Apply light Gaussian smoothing to reduce artifacts
    from scipy.ndimage import gaussian_filter
    print("Applying light smoothing to reduce reconstruction artifacts...")
    reconstructed_map = gaussian_filter(reconstructed_map, sigma=0.1)

    # Add this safety check:
    print(f"Checking reconstruction for anomalies...")
    print(f"  Reconstructed range: [{reconstructed_map.min():.2f}, {reconstructed_map.max():.2f}]")

    # Clamp immediately if values are unreasonable
    if abs(reconstructed_map.min()) > 200 or abs(reconstructed_map.max()) > 200:
        print(f"  WARNING: Extreme values detected! Clamping to safe range.")
        reconstructed_map = np.clip(reconstructed_map, -150, 150)

    # 5. Stitching: North (Truth) + South (Predicted)
    print("Stitching North (Truth) and South (Predicted)...")
    final_map = np.zeros_like(reconstructed_map)
    
    height = final_map.shape[0]
    equator = height // 2
    
    # TOP half: Ground truth (MESSENGER data)
    final_map[:equator, :] = grav_truth[:equator, :]

    # BOTTOM half: AI reconstruction
    final_map[equator:, :] = reconstructed_map[equator:, :]

    # --- ADD THIS NEW SECTION HERE ---
    #print("Clamping values to range [-70, 60] mGal...")
    # This forces any number lower than -70 to become -70
    # And any number higher than 60 to become 60
    #final_map = np.clip(final_map, -70, 60)
    
    # Also clamp the ground truth and input for fair comparison in the plot
    #grav_truth = np.clip(grav_truth, -70, 60)
    #grav_low = np.clip(grav_low, -70, 60)

    # ---------------------------------
    
    # Optional: Smooth the seam at the equator
    # (Simple blend over 10 pixels)
    blend_width = 10
    for i in range(blend_width):
        alpha = i / blend_width
        row = equator - blend_width // 2 + i
        if 0 <= row < height:
            final_map[row, :] = (1 - alpha) * grav_truth[row, :] + alpha * reconstructed_map[row, :]

    # 6. Visualization
    print("Saving results...")
    
    # Upscale low-res input for better visualization (smoother appearance)
    from scipy.ndimage import zoom
    upscale_factor = 4  # Make it 4x larger for smoother look
    grav_low_upscaled = zoom(grav_low, upscale_factor, order=3)  # order=3 = bicubic
    print(f"Upscaled low-res for visualization: {grav_low.shape} -> {grav_low_upscaled.shape}")
    
    # Also upscale ground truth for fair comparison
    grav_truth_upscaled = zoom(grav_truth, upscale_factor, order=3)
    print(f"Upscaled ground truth for visualization: {grav_truth.shape} -> {grav_truth_upscaled.shape}")
    
    # Upscale final map too
    final_map_upscaled = zoom(final_map, upscale_factor, order=3)
    print(f"Upscaled final map for visualization: {final_map.shape} -> {final_map_upscaled.shape}")
    
    # Calculate actual data range for better visualization
    # Use original data range before upscaling
    data_min = min(grav_low.min(), grav_truth.min(), final_map.min())
    data_max = max(grav_low.max(), grav_truth.max(), final_map.max())
    print(f"Data range before upscaling: [{data_min:.2f}, {data_max:.2f}] mGal")

    # Clamp to reasonable range for visualization
    data_min = max(data_min, -150)
    data_max = min(data_max, 150)
    print(f"Clamped visualization range: [{data_min:.2f}, {data_max:.2f}] mGal")
    
    plt.figure(figsize=(15, 10))
    
    # UPDATED SETTINGS: Fixed range -100 to 100 for clearer contrast
    #data_min = min(grav_low.min(), grav_truth.min(), final_map.min())
    #data_max = max(grav_low.max(), grav_truth.max(), final_map.max())
    #print(f"Data range: [{data_min:.2f}, {data_max:.2f}] mGal")

    # Use the actual range for visualization
    viz_args = {'cmap': 'RdBu_r', 'vmin': -150, 'vmax': 150}

    plt.subplot(3, 1, 1)
    plt.title("Original Low-Res Input (L25) - Upscaled for Visualization")
    plt.imshow(grav_low_upscaled, **viz_args, interpolation='bilinear')
    plt.colorbar(label="mGal")
    plt.ylabel('Latitude (degrees)')
    plt.yticks(ticks=[0, 50, 100, 150, 200], labels=['-90', '-45', '0', '45', '90'])

    plt.subplot(3, 1, 2)
    plt.title("Ground Truth (North Only) - Upscaled for Visualization")
    plt.imshow(grav_truth_upscaled, **viz_args, interpolation='bilinear')
    plt.colorbar(label="mGal")
    plt.ylabel('Latitude (degrees)')
    plt.yticks(ticks=[0, 50, 100, 150, 200], labels=['-90', '-45', '0', '45', '90'])

    plt.subplot(3, 1, 3)
    plt.title("Final Hybrid Map: Truth (North) + AI Reconstruction (South)")
    plt.imshow(final_map_upscaled, **viz_args, interpolation='bilinear')
    plt.colorbar(label="mGal")
    plt.ylabel('Latitude (degrees)')
    plt.yticks(ticks=[0, 50, 100, 150, 200], labels=['-90', '-45', '0', '45', '90'])
    
    plt.tight_layout()
    
    # Save visualization as PNG
    viz_path = f"{args.output_dir}/mercury_reconstruction_comparison_3.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to {viz_path}")
    
    # Save raw data
    np.save(f"{args.output_dir}/mercury_final_hybrid_map_3.npy", final_map)
    print(f"Saved hybrid map to {args.output_dir}/mercury_final_hybrid_map_3.npy")

if __name__ == "__main__":
    main()