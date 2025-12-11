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

from moon_model import GravityReconstructionNetwork, DEMRefiningNetwork
from step3_train_moon_base import (
    configure_tensorflow_for_cpu,
    resize_grid_to_match,
    create_aligned_patches_with_dem,
    plot_training_history
)

CUSTOM_OBJECTS = {
    'GravityReconstructionNetwork': GravityReconstructionNetwork,
    'DEMRefiningNetwork': DEMRefiningNetwork
}


def load_pretrained_model(model_path):
    """Load pre-trained model with custom objects"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found: {model_path}")

    try:
        with keras.utils.custom_object_scope(CUSTOM_OBJECTS):
            return keras.models.load_model(model_path)
    except:
        return keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)


def load_and_align_data(grav_low_path, grav_high_path, dem_path):
    """Load gravity and DEM data, ensuring all arrays match in shape"""
    gravity_low = np.load(grav_low_path)
    gravity_high = np.load(grav_high_path)
    dem_high = np.load(dem_path)

    if gravity_low.shape != gravity_high.shape:
        gravity_low = resize_grid_to_match(gravity_low, gravity_high.shape)
    if dem_high.shape != gravity_high.shape:
        dem_high = resize_grid_to_match(dem_high, gravity_high.shape)

    return gravity_low, gravity_high, dem_high


def split_hemisphere(arrays, use_north=True):
    """Split arrays at equator, returning specified hemisphere"""
    midpoint = arrays[0].shape[0] // 2
    if use_north:
        return tuple(arr[:midpoint, :] for arr in arrays)
    return tuple(arr[midpoint:, :] for arr in arrays)


def load_normalization_params(model_dir, fallback_patches=None):
    """Load Moon normalization parameters, or compute from patches as fallback"""
    norm_path = os.path.join(model_dir, 'normalization_params_dem.npz')

    if os.path.exists(norm_path):
        print("  Loading Moon normalization params (Transfer Learning)")
        params = np.load(norm_path)
        return {
            'low_mean': params['low_mean'],
            'low_std': params['low_std'],
            'high_mean': params['high_mean'],
            'high_std': params['high_std'],
            'dem_mean': params['dem_mean'],
            'dem_std': params['dem_std']
        }

    if fallback_patches:
        print("  WARNING: Using Mercury stats (Moon params not found)")
        low_p, high_p, dem_p = fallback_patches
        return {
            'low_mean': np.mean(low_p),
            'low_std': np.std(low_p),
            'high_mean': np.mean(high_p),
            'high_std': np.std(high_p),
            'dem_mean': np.mean(dem_p),
            'dem_std': np.std(dem_p)
        }

    raise ValueError("Cannot determine normalization parameters")


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

    print(f"\nLoading pre-trained Moon model from {moon_model_path}...")
    model = load_pretrained_model(moon_model_path)
    print(f"  Model loaded successfully")
    print(f"  Total parameters: {model.count_params():,}")

    print(f"\nLoading Mercury data (L{l_low} -> L{l_high})...")
    gravity_low, gravity_high, dem_high = load_and_align_data(
        mercury_grav_low_path,
        mercury_grav_high_path,
        mercury_dem_high_path
    )


    print("\nAPPLYING HEMISPHERIC MASK: Training on hemisphere with MESSENGER coverage")
    gravity_low_north, gravity_high_north, dem_high_north = split_hemisphere(
        [gravity_low, gravity_high, dem_high],
        use_north=True
    )

    print(f"Original Shape: {gravity_low.shape} (Global)")
    print(f"Training Shape: {gravity_low_north.shape} (North Only)")

    print("\nCreating aligned patch triplets...")
    low_patches, high_patches, dem_patches = create_aligned_patches_with_dem(
        gravity_low_north,
        gravity_high_north,
        dem_high_north,
        patch_size=30,
        stride=5,
        augment=True
    )
    print(f"Generated {len(low_patches)} patches from Northern Hemisphere.")

    print("\nNormalizing...")
    norm_params = load_normalization_params(
        os.path.dirname(moon_model_path),
        fallback_patches=(low_patches, high_patches, dem_patches)
    )
    low_mean, low_std = norm_params['low_mean'], norm_params['low_std']
    high_mean, high_std = norm_params['high_mean'], norm_params['high_std']
    dem_mean, dem_std = norm_params['dem_mean'], norm_params['dem_std']

    X_gravity = (low_patches - low_mean) / (low_std + 1e-8)
    X_dem = (dem_patches - dem_mean) / (dem_std + 1e-8)
    y = (high_patches - high_mean) / (high_std + 1e-8)

    split_idx = int(0.8 * len(y))
    indices = np.random.permutation(len(y))

    X_gravity_train = X_gravity[indices[:split_idx]]
    X_dem_train = X_dem[indices[:split_idx]]
    y_train = y[indices[:split_idx]]

    X_gravity_val = X_gravity[indices[split_idx:]]
    X_dem_val = X_dem[indices[split_idx:]]
    y_val = y[indices[split_idx:]]

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