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

# Import from existing scripts
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
    epochs=50,  # Fewer epochs for fine-tuning
    batch_size=16,  # Smaller batch for limited Mercury data
    initial_lr=1e-5,  # Much lower learning rate for fine-tuning
    patience=15,
    save_dir='checkpoints_mercury'
):
    """
    Fine-tune pre-trained Moon model on Mercury data

    This implements Phase 2 of the paper's training strategy:
    - Load model pre-trained on Lunar data
    - Fine-tune on limited Mercury data with lower learning rate
    - Adapt to Mercury's specific crustal density/porosity characteristics

    Args:
        moon_model_path: Path to pre-trained Moon model
        mercury_grav_low_path: Path to Mercury low-res gravity
        mercury_grav_high_path: Path to Mercury high-res gravity
        mercury_dem_high_path: Path to Mercury high-res DEM
        l_low: Low resolution degree (25)
        l_high: High resolution degree (50-100)
        epochs: Maximum fine-tuning epochs (30-50)
        batch_size: Batch size (16 for small Mercury dataset)
        initial_lr: Initial learning rate (1e-5, 20x lower than pre-training)
        patience: Early stopping patience (15)
        save_dir: Directory to save fine-tuned models
    """
    print("\n" + "="*80)
    print("MERCURY FINE-TUNING - PHASE 2")
    print("Transfer Learning from Pre-trained Moon Model")
    print("="*80)

    configure_tensorflow_for_cpu()
    os.makedirs(save_dir, exist_ok=True)


    # LOAD PRE-TRAINED MODEL
    print(f"\nLoading pre-trained Moon model from {moon_model_path}...")

    if not os.path.exists(moon_model_path):
        print(f"\nERROR: Pre-trained model not found: {moon_model_path}")
        print("Please run Phase 1 training first:")
        print("  python train_with_dem.py --l_low 25 --l_high 200 --epochs 150")
        return None, None

    model = keras.models.load_model(moon_model_path)
    print(f"  Model loaded successfully")
    print(f"  Total parameters: {model.count_params():,}")

    # LOAD MERCURY DATA
    print(f"\nLoading Mercury data (L{l_low} -> L{l_high})...")
    start_time = time.time()

    # Check if files exist
    for path in [mercury_grav_low_path, mercury_grav_high_path, mercury_dem_high_path]:
        if not os.path.exists(path):
            print(f"\nERROR: File not found: {path}")
            print("\nMercury data preparation needed:")
            print("1. Download MESSENGER HgM007 gravity model")
            print("2. Process Mercury DEM from MDIS")
            print("3. Run preprocessing pipeline")
            print("\nSee DEM_TRAINING_GUIDE.md for details")
            return None, None

    gravity_low = np.load(mercury_grav_low_path)
    gravity_high = np.load(mercury_grav_high_path)
    dem_high = np.load(mercury_dem_high_path)

    print(f"  Low-res gravity: {gravity_low.shape}")
    print(f"  High-res gravity: {gravity_high.shape}")
    print(f"  High-res DEM: {dem_high.shape}")
    print(f"  Loading time: {time.time() - start_time:.2f}s")

    # Verify DEM-Gravity spatial similarity for Mercury
    from dem_preprocessing import calculate_dem_gravity_ssim
    ssim_value = calculate_dem_gravity_ssim(dem_high, gravity_high)


    # CREATE PATCHES
    print("\nCreating aligned patch triplets...")
    print("  Note: Limited Mercury data, no augmentation to prevent overfitting")
    start_time = time.time()

    # Use smaller stride for better coverage with limited data
    low_patches, high_patches, dem_patches = create_aligned_patches_with_dem(
        gravity_low, gravity_high, dem_high,
        patch_size=30,
        stride=15,  # More overlap for limited data
        augment=False  # No augmentation to avoid overfitting
    )

    print(f"  Patch creation time: {time.time() - start_time:.2f}s")
    print(f"  Mercury dataset size: {len(low_patches)} patches")

    if len(low_patches) < 100:
        print("\n  WARNING: Very small dataset! Consider:")
        print("    - Using smaller stride (stride=10)")
        print("    - Enabling augmentation (augment=True)")
        print("    - Using more Mercury gravity data if available")

    # ========================================================================
    # NORMALIZE (using Moon statistics for transfer learning)
    # ========================================================================
    print("\nNormalizing using Moon statistics (transfer learning)...")

    # Load Moon normalization parameters
    moon_norm_path = os.path.join(
        os.path.dirname(moon_model_path),
        'normalization_params_dem.npz'
    )

    if os.path.exists(moon_norm_path):
        print(f"  Loading Moon normalization params from {moon_norm_path}")
        moon_norm = np.load(moon_norm_path)
        low_mean = moon_norm['low_mean']
        low_std = moon_norm['low_std']
        high_mean = moon_norm['high_mean']
        high_std = moon_norm['high_std']
        dem_mean = moon_norm['dem_mean']
        dem_std = moon_norm['dem_std']
    else:
        print("  WARNING: Moon normalization params not found")
        print("  Computing normalization from Mercury data...")
        low_mean, low_std = np.mean(low_patches), np.std(low_patches)
        high_mean, high_std = np.mean(high_patches), np.std(high_patches)
        dem_mean, dem_std = np.mean(dem_patches), np.std(dem_patches)

    print(f"    Gravity mean: {low_mean:.4f}, std: {low_std:.4f}")
    print(f"    DEM mean: {dem_mean:.4f}, std: {dem_std:.4f}")

    # Save Mercury-specific normalization parameters
    np.savez(
        f'{save_dir}/normalization_params_mercury.npz',
        low_mean=low_mean, low_std=low_std,
        high_mean=high_mean, high_std=high_std,
        dem_mean=dem_mean, dem_std=dem_std
    )

    X_gravity = (low_patches - low_mean) / (low_std + 1e-8)
    X_dem = (dem_patches - dem_mean) / (dem_std + 1e-8)
    y = (high_patches - high_mean) / (high_std + 1e-8)

    # ========================================================================
    # TRAIN/VAL SPLIT
    # ========================================================================
    print("\nSplitting train/validation (80/20)...")
    split_idx = int(0.8 * len(y))
    indices = np.random.permutation(len(y))

    X_gravity_train = X_gravity[indices[:split_idx]]
    X_dem_train = X_dem[indices[:split_idx]]
    y_train = y[indices[:split_idx]]

    X_gravity_val = X_gravity[indices[split_idx:]]
    X_dem_val = X_dem[indices[split_idx:]]
    y_val = y[indices[split_idx:]]

    print(f"  Training: {len(y_train)} patches")
    print(f"  Validation: {len(y_val)} patches")

    # ========================================================================
    # RECOMPILE WITH LOWER LEARNING RATE
    # ========================================================================
    print(f"\nRecompiling model for fine-tuning...")
    print(f"  New learning rate: {initial_lr} (20x lower than pre-training)")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss='mae',
        metrics=['mae', 'mse']
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'{save_dir}/mercury_model_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,  # More aggressive for fine-tuning
            min_lr=1e-8,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(f'{save_dir}/finetuning_log.csv'),
        keras.callbacks.TensorBoard(
            log_dir=f'{save_dir}/logs',
            histogram_freq=0,
            write_graph=False
        )
    ]


    # FINE-TUNE

    print("\n" + "="*80)
    print("STARTING FINE-TUNING ON MERCURY DATA")
    print("="*80)
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Early stopping patience: {patience}")
    print(f"  DEM-Gravity SSIM: {ssim_value:.4f}")
    print("="*80 + "\n")

    start_time = time.time()

    history = model.fit(
        [X_gravity_train, X_dem_train], y_train,
        validation_data=([X_gravity_val, X_dem_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    training_time = time.time() - start_time

    # Save final model
    model.save(f'{save_dir}/mercury_model_final.h5')

    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"  Total fine-tuning time: {training_time/60:.2f} minutes")
    print(f"  Best val_loss: {min(history.history['val_loss']):.4f}")
    print(f"  Final val_loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Epochs trained: {len(history.history['loss'])}")
    print(f"  Mercury DEM-Gravity SSIM: {ssim_value:.4f}")
    print("="*80)

    # Plot training curves
    plot_training_history(history, save_dir)

    return model, history


def compare_moon_vs_mercury_models(
    moon_model_path,
    mercury_model_path,
    mercury_test_grav_low,
    mercury_test_grav_high,
    mercury_test_dem_high,
    save_dir='results_mercury_comparison'
):
    """
    Compare Moon pre-trained model vs Mercury fine-tuned model on Mercury test data

    Args:
        moon_model_path: Pre-trained Moon model
        mercury_model_path: Fine-tuned Mercury model
        mercury_test_*: Mercury test data paths
        save_dir: Output directory
    """
    from evaluate_dem_model import (
        load_model_and_data,
        predict_full_field,
        calculate_metrics,
        visualize_results
    )

    print("\n" + "="*80)
    print("COMPARING MOON PRE-TRAINED vs MERCURY FINE-TUNED")
    print("="*80)

    # Load data
    _, grav_low, grav_high, dem_high = load_model_and_data(
        moon_model_path,
        mercury_test_grav_low,
        mercury_test_grav_high,
        mercury_test_dem_high
    )

    # Test Moon model on Mercury
    print("\n--- MOON PRE-TRAINED MODEL (Zero-shot Transfer) ---")
    moon_model = keras.models.load_model(moon_model_path)
    pred_moon = predict_full_field(moon_model, grav_low, dem_high=dem_high)
    metrics_moon = calculate_metrics(pred_moon, grav_high)

    # Test Mercury fine-tuned model
    print("\n--- MERCURY FINE-TUNED MODEL ---")
    mercury_model = keras.models.load_model(mercury_model_path)
    pred_mercury = predict_full_field(mercury_model, grav_low, dem_high=dem_high)
    metrics_mercury = calculate_metrics(pred_mercury, grav_high)

    # Comparison
    print("\n" + "="*80)
    print("MERCURY FINE-TUNING IMPROVEMENT")
    print("="*80)
    print(f"{'Metric':<25s} {'Moon Model':>15s} {'Mercury Model':>15s} {'Improvement':>15s}")
    print("-"*80)

    for key in metrics_moon.keys():
        val_m = metrics_moon[key]
        val_hg = metrics_mercury[key]

        if 'Correlation' in key or 'SSIM' in key:
            improvement = ((val_hg - val_m) / val_m) * 100
        else:
            improvement = ((val_m - val_hg) / val_m) * 100

        print(f"{key:<25s} {val_m:>15.6f} {val_hg:>15.6f} {improvement:>14.2f}%")

    print("="*80)

    # Visualizations
    os.makedirs(save_dir, exist_ok=True)

    visualize_results(grav_low, pred_moon, grav_high, dem_high,
                     save_dir=f'{save_dir}/moon_zero_shot')
    visualize_results(grav_low, pred_mercury, grav_high, dem_high,
                     save_dir=f'{save_dir}/mercury_finetuned')

    return metrics_moon, metrics_mercury


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune Moon model on Mercury data')
    parser.add_argument('--moon_model', type=str,
                       default='checkpoints_dem/moon_full_model_best.h5',
                       help='Path to pre-trained Moon model')
    parser.add_argument('--mercury_grav_low', type=str,
                       default='data/processed/mercury_grav_L25.npy',
                       help='Mercury low-res gravity')
    parser.add_argument('--mercury_grav_high', type=str,
                       default='data/processed/mercury_grav_L50.npy',
                       help='Mercury high-res gravity')
    parser.add_argument('--mercury_dem_high', type=str,
                       default='data/processed/mercury_dem_high.npy',
                       help='Mercury high-res DEM')
    parser.add_argument('--l_low', type=int, default=25,
                       help='Low resolution degree')
    parser.add_argument('--l_high', type=int, default=50,
                       help='High resolution degree')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (much lower than pre-training)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--compare', action='store_true',
                       help='Compare Moon vs Mercury models after fine-tuning')

    args = parser.parse_args()

    print("="*80)
    print("MERCURY FINE-TUNING - PHASE 2 TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Pre-trained model: {args.moon_model}")
    print(f"  Low degree: L={args.l_low}")
    print(f"  High degree: L={args.l_high}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr} (fine-tuning)")
    print(f"  Patience: {args.patience}")

    # Fine-tune
    model, history = finetune_mercury_model(
        moon_model_path=args.moon_model,
        mercury_grav_low_path=args.mercury_grav_low,
        mercury_grav_high_path=args.mercury_grav_high,
        mercury_dem_high_path=args.mercury_dem_high,
        l_low=args.l_low,
        l_high=args.l_high,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        patience=args.patience
    )

    if model is not None:
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Check fine-tuning curves: checkpoints_mercury/finetuning_log.csv")
        print("2. Fine-tuned model: checkpoints_mercury/mercury_model_best.h5")
        print("3. Apply to Mercury southern hemisphere reconstruction")

        if args.compare:
            print("4. Running comparison analysis...")
            compare_moon_vs_mercury_models(
                args.moon_model,
                'checkpoints_mercury/mercury_model_best.h5',
                args.mercury_grav_low,
                args.mercury_grav_high,
                args.mercury_dem_high
            )

        print("="*80)
