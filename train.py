"""
CPU-Optimized Training Script
Faster training with mixed precision, larger batches, and efficient data loading
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time

from model import create_gravity_only_model


def configure_tensorflow_for_cpu():
    """Optimize TensorFlow for CPU training"""
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.optimizer.set_jit(True)


def resize_grid_to_match(grid_small, target_shape):
    """Resize smaller grid to match larger grid using fast interpolation"""
    zoom_factors = (target_shape[0] / grid_small.shape[0],
                    target_shape[1] / grid_small.shape[1])
    return zoom(grid_small, zoom_factors, order=1)


def create_aligned_patches(grid_low, grid_high, patch_size=30, stride=20, augment=True):
    """
    Create spatially aligned patch pairs
    Optimized: uses stride=20 for fewer but less correlated patches
    """
    if grid_low.shape != grid_high.shape:
        grid_low = resize_grid_to_match(grid_low, grid_high.shape)

    h, w = grid_high.shape
    low_patches, high_patches = [], []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            low_patch = grid_low[i:i+patch_size, j:j+patch_size]
            high_patch = grid_high[i:i+patch_size, j:j+patch_size]

            if np.isnan(low_patch).any() or np.isnan(high_patch).any():
                continue

            low_patches.append(low_patch)
            high_patches.append(high_patch)

            if augment:
                low_patches.append(np.fliplr(low_patch))
                high_patches.append(np.fliplr(high_patch))
                low_patches.append(np.flipud(low_patch))
                high_patches.append(np.flipud(high_patch))

    low_patches = np.array(low_patches, dtype=np.float32)[..., np.newaxis]
    high_patches = np.array(high_patches, dtype=np.float32)[..., np.newaxis]
    return low_patches, high_patches


def train_moon_gravity_model(
    l_low=25,
    l_high=200,
    epochs=150,
    batch_size=32,
    initial_lr=2e-4,
    patience=20,
    save_dir='checkpoints'
):
    """
    Train gravity reconstruction model on Moon data (CPU-optimized)

    Args:
        l_low: Low resolution degree (25)
        l_high: High resolution degree (200)
        epochs: Maximum training epochs (150)
        batch_size: Batch size (32 = good for CPU)
        initial_lr: Initial learning rate (2e-4 = slightly higher for speed)
        patience: Early stopping patience (20)
        save_dir: Directory to save models
    """
    configure_tensorflow_for_cpu()
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    grid_low = np.load(f"data/processed/moon_grav_L{l_low}.npz")['grid']
    grid_high = np.load(f"data/processed/moon_grav_L{l_high}.npz")['grid']

    low_patches, high_patches = create_aligned_patches(
        grid_low, grid_high,
        patch_size=30,
        stride=20,
        augment=True
    )

    low_mean, low_std = np.mean(low_patches), np.std(low_patches)
    high_mean, high_std = np.mean(high_patches), np.std(high_patches)

    np.savez(
        f'{save_dir}/normalization_params.npz',
        low_mean=low_mean, low_std=low_std,
        high_mean=high_mean, high_std=high_std
    )

    X = (low_patches - low_mean) / (low_std + 1e-8)
    y = (high_patches - high_mean) / (high_std + 1e-8)

    split_idx = int(0.8 * len(y))
    indices = np.random.permutation(len(y))

    X_train = X[indices[:split_idx]]
    y_train = y[indices[:split_idx]]
    X_val = X[indices[split_idx:]]
    y_val = y[indices[split_idx:]]

    model = create_gravity_only_model(
        patch_size=30,
        growth_rate=12,
        num_blocks=4,
        dropout_rate=0.2,
        l2_reg=1e-5
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'{save_dir}/moon_gravity_model_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(f'{save_dir}/training_log.csv'),
        keras.callbacks.TensorBoard(
            log_dir=f'{save_dir}/logs',
            histogram_freq=0,
            write_graph=False
        )
    ]

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    training_time = time.time() - start_time

    model.save(f'{save_dir}/moon_gravity_model_final.h5')

    plot_training_history(history, save_dir)

    return model, history


def plot_training_history(history, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('MSE Loss', fontsize=11)
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # MSE
    axes[0, 1].plot(history.history['mse'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_mse'], label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('MSE', fontsize=11)
    axes[0, 1].set_title('MSE Curves', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Log scale loss
    axes[1, 0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('MSE Loss (log scale)', fontsize=11)
    axes[1, 0].set_title('Loss Curves (Log Scale)', fontsize=12, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        train_loss = np.array(history.history['loss'])
        val_loss = np.array(history.history['val_loss'])
        gap = val_loss - train_loss
        axes[1, 1].plot(gap, linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Val Loss - Train Loss', fontsize=11)
        axes[1, 1].set_title('Generalization Gap', fontsize=12, fontweight='bold')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train gravity reconstruction model (CPU-optimized)')
    parser.add_argument('--l_low', type=int, default=25, help='Low resolution degree')
    parser.add_argument('--l_high', type=int, default=200, help='High resolution degree')
    parser.add_argument('--epochs', type=int, default=150, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    args = parser.parse_args()

    model, history = train_moon_gravity_model(
        l_low=args.l_low,
        l_high=args.l_high,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        patience=args.patience
    )
