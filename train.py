import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import argparse
from datetime import datetime

from model import create_model, create_gravity_only_model
from model_eval import MetricsCallback, evaluate_model


class LearningRateScheduler(keras.callbacks.Callback):
    """halve learning rate every 100 epochs"""
    def __init__(self, initial_lr=1e-4, decay_epochs=100):
        super(LearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        num_decays = epoch // self.decay_epochs
        new_lr = self.initial_lr * (0.5 ** num_decays)
        keras.backend.set_value(self.model.optimizer.lr, new_lr)
        if epoch % self.decay_epochs == 0 and epoch > 0:
            print(f"\nLearning rate adjusted to {new_lr:.6f} at epoch {epoch}")


def create_patch_dataset(gravity_field, patch_size=30, stride=15):
    """create 30x30 patches from gravity field with sliding window"""
    if len(gravity_field.shape) == 3 and gravity_field.shape[-1] == 1:
        gravity_field = gravity_field[:, :, 0]

    h, w = gravity_field.shape
    patches = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = gravity_field[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=-1)
    return patches


def prepare_training_pairs(low_res_field, high_res_field, dem_field=None,
                           patch_size=30, stride=15, normalize=True):
    """prepare training pairs from low-res and high-res gravity fields"""
    low_res_patches = create_patch_dataset(low_res_field, patch_size, stride)
    high_res_patches = create_patch_dataset(high_res_field, patch_size, stride)

    if normalize:
        low_mean, low_std = np.mean(low_res_patches), np.std(low_res_patches)
        high_mean, high_std = np.mean(high_res_patches), np.std(high_res_patches)
        low_res_patches = (low_res_patches - low_mean) / (low_std + 1e-8)
        high_res_patches = (high_res_patches - high_mean) / (high_std + 1e-8)

    if dem_field is not None:
        dem_patches = create_patch_dataset(dem_field, patch_size, stride)
        if normalize:
            dem_mean, dem_std = np.mean(dem_patches), np.std(dem_patches)
            dem_patches = (dem_patches - dem_mean) / (dem_std + 1e-8)
        return ([low_res_patches, dem_patches], high_res_patches)
    else:
        return (low_res_patches, high_res_patches)


def train_gravity_network(train_data, train_labels, val_data, val_labels,
                          epochs=300, batch_size=32, initial_lr=1e-4,
                          save_dir='checkpoints', model_name='gravity_net'):
    """train the gravity reconstruction network"""
    print("="*80)
    print("training gravity reconstruction network")
    print("="*80)

    model = create_gravity_only_model(patch_size=30, growth_rate=16, num_blocks=6)

    optimizer = keras.optimizers.Adam(
        learning_rate=initial_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=1e-4
    )

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        LearningRateScheduler(initial_lr=initial_lr, decay_epochs=100),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', model_name, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        ),
        MetricsCallback(validation_data=(val_data, val_labels), log_frequency=5)
    ]

    print(f"\ntraining on {len(train_data)} samples")
    print(f"validation on {len(val_data)} samples")
    print(f"batch size: {batch_size}")
    print(f"initial learning rate: {initial_lr}")
    print(f"weight decay: 1e-4\n")

    history = model.fit(
        train_data, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(save_dir, f'{model_name}_final.h5'))
    print(f"\nmodel saved to {save_dir}/{model_name}_final.h5")
    return model, history


def train_complete_model(train_data, train_labels, val_data, val_labels,
                        epochs=300, batch_size=32, initial_lr=1e-4,
                        save_dir='checkpoints', model_name='complete_model',
                        pretrained_gravity_weights=None):
    """train the complete model with dem refinement"""
    print("="*80)
    print("training complete model")
    print("="*80)

    model = create_model(patch_size=30, growth_rate=16, num_blocks=6)

    if pretrained_gravity_weights is not None:
        print(f"\nloading pretrained weights from {pretrained_gravity_weights}")
        try:
            pretrained_model = keras.models.load_model(pretrained_gravity_weights, compile=False)
            for layer in model.layers:
                if 'gravity' in layer.name.lower():
                    try:
                        layer.set_weights(pretrained_model.get_layer(layer.name).get_weights())
                    except:
                        pass
            print("pretrained weights loaded")
        except Exception as e:
            print(f"warning: could not load pretrained weights: {e}")

    optimizer = keras.optimizers.Adam(
        learning_rate=initial_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=1e-4
    )

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        LearningRateScheduler(initial_lr=initial_lr, decay_epochs=100),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', model_name, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        ),
        MetricsCallback(validation_data=(val_data, val_labels), log_frequency=5)
    ]

    print(f"\ntraining on {len(train_labels)} samples")
    print(f"validation on {len(val_labels)} samples")
    print(f"batch size: {batch_size}")
    print(f"initial learning rate: {initial_lr}\n")

    history = model.fit(
        train_data, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(save_dir, f'{model_name}_final.h5'))
    print(f"\nmodel saved to {save_dir}/{model_name}_final.h5")
    return model, history


def main():
    """main training function with command line arguments"""
    parser = argparse.ArgumentParser(description='train mercury gravity reconstruction model')

    parser.add_argument('--mode', type=str, default='gravity_only',
                       choices=['gravity_only', 'complete'],
                       help='Training mode: gravity_only or complete (with DEM)')

    parser.add_argument('--train_low', type=str, required=True,
                       help='Path to low-resolution training data (.npy)')

    parser.add_argument('--train_high', type=str, required=True,
                       help='Path to high-resolution training data (.npy)')

    parser.add_argument('--train_dem', type=str, default=None,
                       help='Path to DEM training data (.npy, required for complete mode)')

    parser.add_argument('--val_low', type=str, default=None,
                       help='Path to low-resolution validation data (.npy)')

    parser.add_argument('--val_high', type=str, default=None,
                       help='Path to high-resolution validation data (.npy)')

    parser.add_argument('--val_dem', type=str, default=None,
                       help='Path to DEM validation data (.npy)')

    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs (default: 300)')

    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')

    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate (default: 1e-4)')

    parser.add_argument('--patch_size', type=int, default=30,
                       help='Patch size (default: 30)')

    parser.add_argument('--stride', type=int, default=15,
                       help='Stride for patch extraction (default: 15)')

    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='Path to pretrained gravity network weights (for complete mode)')

    args = parser.parse_args()

    print("loading training data...")
    train_low_field = np.load(args.train_low)
    train_high_field = np.load(args.train_high)

    if args.mode == 'gravity_only':
        train_data, train_labels = prepare_training_pairs(
            train_low_field, train_high_field,
            patch_size=args.patch_size, stride=args.stride
        )
    else:
        if args.train_dem is None:
            raise ValueError("dem data required for complete mode")
        train_dem_field = np.load(args.train_dem)
        train_data, train_labels = prepare_training_pairs(
            train_low_field, train_high_field, train_dem_field,
            patch_size=args.patch_size, stride=args.stride
        )

    if args.val_low is not None and args.val_high is not None:
        print("loading validation data...")
        val_low_field = np.load(args.val_low)
        val_high_field = np.load(args.val_high)

        if args.mode == 'gravity_only':
            val_data, val_labels = prepare_training_pairs(
                val_low_field, val_high_field,
                patch_size=args.patch_size, stride=args.stride
            )
        else:
            val_dem_field = np.load(args.val_dem)
            val_data, val_labels = prepare_training_pairs(
                val_low_field, val_high_field, val_dem_field,
                patch_size=args.patch_size, stride=args.stride
            )
    else:
        split_idx = int(0.8 * len(train_labels))
        if args.mode == 'gravity_only':
            val_data = train_data[split_idx:]
            train_data = train_data[:split_idx]
        else:
            val_data = [train_data[0][split_idx:], train_data[1][split_idx:]]
            train_data = [train_data[0][:split_idx], train_data[1][:split_idx]]

        val_labels = train_labels[split_idx:]
        train_labels = train_labels[:split_idx]

    if args.mode == 'gravity_only':
        model, history = train_gravity_network(
            train_data, train_labels, val_data, val_labels,
            epochs=args.epochs, batch_size=args.batch_size,
            initial_lr=args.lr, save_dir=args.save_dir
        )
    else:
        model, history = train_complete_model(
            train_data, train_labels, val_data, val_labels,
            epochs=args.epochs, batch_size=args.batch_size,
            initial_lr=args.lr, save_dir=args.save_dir,
            pretrained_gravity_weights=args.pretrained_weights
        )

    print("\n" + "="*80)
    print("final evaluation")
    print("="*80)
    results = evaluate_model(model, val_data, val_labels)
    print("\ntraining complete!")


if __name__ == "__main__":
    main()
