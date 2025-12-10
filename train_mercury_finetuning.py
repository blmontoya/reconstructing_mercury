"""
transfer learning from pre-trained Moon model to Mercury data
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time

# IMPORT CUSTOM CLASSES (REQUIRED FOR LOADING)
from model import GravityReconstructionNetwork, DEMRefiningNetwork

# Import from existing scripts
# (Ensure train_with_dem.py exists and has these functions, or copy them here)
from train_with_dem import (
    configure_tensorflow_for_cpu,
    resize_grid_to_match,
    create_aligned_patches_with_dem,
    plot_training_history
)

def finetune_mercury_model(
    moon_model_path,
    mercury_grav_low_path,
    mercury_grav_high_path,
    mercury_dem_high_path,
    l_low=25,
    l_high=50, 
    epochs=50,
    batch_size=16,
    initial_lr=1e-5,
    patience=15,
    save_dir='checkpoints_mercury'
):
    print("\n" + "="*80)
    print("MERCURY FINE-TUNING - PHASE 2")
    print("Transfer Learning from Pre-trained Moon Model")
    print("="*80)

    configure_tensorflow_for_cpu()
    os.makedirs(save_dir, exist_ok=True)

    # --- FIX: LOAD PRE-TRAINED MODEL WITH CUSTOM OBJECTS ---
    print(f"\nLoading pre-trained Moon model from {moon_model_path}...")

    if not os.path.exists(moon_model_path):
        print(f"\nERROR: Pre-trained model not found: {moon_model_path}")
        return None, None

    # Register custom layers so Keras can read the file
    custom_objects = {
        'GravityReconstructionNetwork': GravityReconstructionNetwork,
        'DEMRefiningNetwork': DEMRefiningNetwork
    }

    try:
        with keras.utils.custom_object_scope(custom_objects):
            model = keras.models.load_model(moon_model_path)
    except:
        # Fallback for older Keras versions
        model = keras.models.load_model(moon_model_path, custom_objects=custom_objects)
        
    print(f"  Model loaded successfully")
    print(f"  Total parameters: {model.count_params():,}")
    # -------------------------------------------------------

# LOAD MERCURY DATA
    print(f"\nLoading Mercury data (L{l_low} -> L{l_high})...")
    
    gravity_low = np.load(mercury_grav_low_path)
    gravity_high = np.load(mercury_grav_high_path)
    dem_high = np.load(mercury_dem_high_path)

    # Resize checks (robustness)
    if gravity_low.shape != gravity_high.shape:
        gravity_low = resize_grid_to_match(gravity_low, gravity_high.shape)
    if dem_high.shape != gravity_high.shape:
        dem_high = resize_grid_to_match(dem_high, gravity_high.shape)


    print("\n" + "!"*60)
    print("APPLYING HEMISPHERIC MASK: Training on hemisphere with MESSENGER coverage")
    print("!"*60)
    
    # Assuming standard map (Row 0 is North Pole, Row H is South Pole)
    # We take the top 50% of the map for training.
    
    height = gravity_low.shape[0]
    midpoint = height // 2  # The equator
    
    # Slice arrays to keep only the hemisphere with real MESSENGER data (TOP half)
    gravity_low_north = gravity_low[:midpoint, :]
    gravity_high_north = gravity_high[:midpoint, :]
    dem_high_north = dem_high[:midpoint, :]
    
    print(f"Original Shape: {gravity_low.shape} (Global)")
    print(f"Training Shape: {gravity_low_north.shape} (North Only)")
    # =================================================================

    # CREATE PATCHES (Now using ONLY Northern Data)
    print("\nCreating aligned patch triplets...")
    low_patches, high_patches, dem_patches = create_aligned_patches_with_dem(
        gravity_low_north,      # <--- Passing sliced data
        gravity_high_north,     # <--- Passing sliced data
        dem_high_north,         # <--- Passing sliced data
        patch_size=30,
        stride=5,
        augment=True 
    )
    
    print(f"Generated {len(low_patches)} patches from Northern Hemisphere.")

    # NORMALIZE (using Moon statistics for transfer learning)
    print("\nNormalizing...")
    
    # Try to load Moon stats to keep input distribution consistent
    moon_norm_path = os.path.join(os.path.dirname(moon_model_path), 'normalization_params_dem.npz')
    
    if os.path.exists(moon_norm_path):
        print(f"  Loading Moon normalization params (Transfer Learning best practice)")
        moon_norm = np.load(moon_norm_path)
        low_mean, low_std = moon_norm['low_mean'], moon_norm['low_std']
        high_mean, high_std = moon_norm['high_mean'], moon_norm['high_std']
        dem_mean, dem_std = moon_norm['dem_mean'], moon_norm['dem_std']
    else:
        print("  WARNING: Moon normalization params not found. Using Mercury stats.")
        low_mean, low_std = np.mean(low_patches), np.std(low_patches)
        high_mean, high_std = np.mean(high_patches), np.std(high_patches)
        dem_mean, dem_std = np.mean(dem_patches), np.std(dem_patches)

    X_gravity = (low_patches - low_mean) / (low_std + 1e-8)
    X_dem = (dem_patches - dem_mean) / (dem_std + 1e-8)
    y = (high_patches - high_mean) / (high_std + 1e-8)

    # TRAIN/VAL SPLIT
    split_idx = int(0.8 * len(y))
    indices = np.random.permutation(len(y))
    
    X_gravity_train = X_gravity[indices[:split_idx]]
    X_dem_train = X_dem[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    
    X_gravity_val = X_gravity[indices[split_idx:]]
    X_dem_val = X_dem[indices[split_idx:]]
    y_val = y[indices[split_idx:]]

    # RECOMPILE WITH LOWER LEARNING RATE
    print(f"\nRecompiling model for fine-tuning (LR: {initial_lr})...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
        loss='mse',
        metrics=['mae', 'mse']
    )

    callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            f'{save_dir}/mercury_model_best.h5',
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-8, verbose=1),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(f'{save_dir}/finetuning_log.csv')
    ]

    # FINE-TUNE
    print("\nSTARTING FINE-TUNING...")
    history = model.fit(
        [X_gravity_train, X_dem_train], y_train,
        validation_data=([X_gravity_val, X_dem_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1,
        shuffle=True
    )

    model.save(f'{save_dir}/mercury_model_final.h5')
    plot_training_history(history, save_dir)
    
    return model, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--moon_model', type=str, default='checkpoints_dem/moon_full_model_best.h5')
    parser.add_argument('--mercury_grav_low', type=str, required=True)
    parser.add_argument('--mercury_grav_high', type=str, required=True)
    parser.add_argument('--mercury_dem_high', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    
    args = parser.parse_args()
    
    finetune_mercury_model(
        moon_model_path=args.moon_model,
        mercury_grav_low_path=args.mercury_grav_low,
        mercury_grav_high_path=args.mercury_grav_high,
        mercury_dem_high_path=args.mercury_dem_high,
        epochs=args.epochs,
        initial_lr=args.lr
    )