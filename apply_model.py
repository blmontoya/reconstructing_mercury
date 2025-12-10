"""
Apply DEM-Enhanced Model
Generates a High-Resolution Gravity Map from Low-Res Gravity + High-Res DEM
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import argparse
from scipy.ndimage import zoom

# IMPORT CUSTOM CLASSES (Required to load the model)
from model import GravityReconstructionNetwork, DEMRefiningNetwork

def load_data(grav_path, dem_path):
    """Load and normalize input data"""
    print(f"Loading input files...")
    print(f"  Gravity Input: {grav_path}")
    print(f"  DEM Input:     {dem_path}")
    
    grav_low = np.load(grav_path)
    dem_high = np.load(dem_path)
    
    # 1. Resize Low-Res Gravity to match High-Res DEM
    # The AI needs both inputs to be the same grid size (pixel-wise)
    if grav_low.shape != dem_high.shape:
        print(f"\n  Upscaling Gravity from {grav_low.shape} to {dem_high.shape}...")
        zoom_factors = (dem_high.shape[0] / grav_low.shape[0], 
                        dem_high.shape[1] / grav_low.shape[1])
        grav_low = zoom(grav_low, zoom_factors, order=1)
    
    # 2. Normalize
    # We use standard normalization (Z-score)
    print("  Normalizing data...")
    g_mean, g_std = np.mean(grav_low), np.std(grav_low)
    d_mean, d_std = np.mean(dem_high), np.std(dem_high)
    
    grav_norm = (grav_low - g_mean) / (g_std + 1e-8)
    dem_norm = (dem_high - d_mean) / (d_std + 1e-8)
    
    # Return normalized data AND stats (so we can un-normalize later)
    return grav_norm, dem_norm, (g_mean, g_std)

def predict_patches(model, grav, dem, patch_size=30, stride=15):
    """Run the model on the full map using sliding windows"""
    print(f"\nGenerating map (Patch Size: {patch_size}, Stride: {stride})...")
    
    h, w = grav.shape
    output = np.zeros_like(grav)
    counts = np.zeros_like(grav)
    
    total_patches = ((h - patch_size) // stride + 1) * ((w - patch_size) // stride + 1)
    processed = 0
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Extract
            g_patch = grav[i:i+patch_size, j:j+patch_size]
            d_patch = dem[i:i+patch_size, j:j+patch_size]
            
            # Batch dimension (1, 30, 30, 1)
            g_input = g_patch[np.newaxis, ..., np.newaxis]
            d_input = d_patch[np.newaxis, ..., np.newaxis]
            
            # Predict
            pred = model.predict([g_input, d_input], verbose=0)
            
            # Accumulate
            output[i:i+patch_size, j:j+patch_size] += pred[0, :, :, 0]
            counts[i:i+patch_size, j:j+patch_size] += 1
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed}/{total_patches} patches...")
                
    # Average overlaps
    return output / (counts + 1e-8)

def main():
    parser = argparse.ArgumentParser(description="Generate High-Res Gravity Map")
    parser.add_argument('--model', type=str, required=True, help='Path to trained .h5 model')
    parser.add_argument('--grav_low', type=str, required=True, help='Path to Low-Res Gravity .npy')
    parser.add_argument('--dem_high', type=str, required=True, help='Path to High-Res DEM .npy')
    parser.add_argument('--output', type=str, default='mercury_reconstructed_L200.npy', help='Output filename')
    parser.add_argument('--visualize', action='store_true', help='Save visualization image')
    args = parser.parse_args()
    
    # 1. Load Data
    grav_norm, dem_norm, (g_mean, g_std) = load_data(args.grav_low, args.dem_high)
    
    # 2. Load Model
    print(f"\nLoading model: {args.model}")
    custom_objects = {
        'GravityReconstructionNetwork': GravityReconstructionNetwork,
        'DEMRefiningNetwork': DEMRefiningNetwork
    }
    with keras.utils.custom_object_scope(custom_objects):
        model = keras.models.load_model(args.model)
        
    # 3. Generate Map
    prediction_norm = predict_patches(model, grav_norm, dem_norm)
    
    # 4. De-normalize to get real mGal units
    prediction_mgal = (prediction_norm * g_std) + g_mean
    
    # 5. Save
    print(f"\nSaving output to {args.output}...")
    np.save(args.output, prediction_mgal)
    
    # 6. Optional Visualization
    if args.visualize:
        viz_file = args.output.replace('.npy', '.png')
        print(f"Saving visualization to {viz_file}...")
        
        plt.figure(figsize=(12, 12))
        
        # Plot 1: Input (Upscaled)
        plt.subplot(3, 1, 1)
        # De-normalize input for fair comparison
        grav_input_mgal = (grav_norm * g_std) + g_mean
        plt.imshow(grav_input_mgal, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='mGal')
        plt.title(f"Input: Low-Resolution Gravity (Upscaled to {grav_norm.shape})")
        
        # Plot 2: Helper DEM
        plt.subplot(3, 1, 2)
        plt.imshow(dem_norm, cmap='terrain', aspect='auto')
        plt.colorbar(label='Normalized Elevation')
        plt.title("Helper: High-Resolution DEM")
        
        # Plot 3: Output
        plt.subplot(3, 1, 3)
        plt.imshow(prediction_mgal, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='mGal')
        plt.title("Output: AI Reconstructed Gravity")
        
        plt.tight_layout()
        plt.savefig(viz_file, dpi=150)
        
    print("\nDone!")

if __name__ == "__main__":
    main()