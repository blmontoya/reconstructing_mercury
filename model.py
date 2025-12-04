import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DenseBlock(layers.Layer):
    """
    Dense block with 6 convolutional layers
    Each layer concatenates all previous layer outputs within the block
    """
    def __init__(self, growth_rate=16, num_layers=6, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.conv_layers = []

        # create 6 convolutional layers with growth rate k=16
        for i in range(num_layers):
            self.conv_layers.append(
                layers.Conv2D(
                    filters=growth_rate,
                    kernel_size=(3, 3),
                    padding='same',  
                    activation='relu',
                    kernel_initializer='he_normal'  
                )
            )

    def call(self, x):
        """
        Forward pass - each layer concatenates all previous outputs
        """
        features = [x]

        for conv_layer in self.conv_layers:
            # concatenate all previous features
            concat_features = layers.concatenate(features, axis=-1)
            # apply convolution
            new_features = conv_layer(concat_features)
            # add to feature list
            features.append(new_features)

        # return concatenation of all features from this block
        return layers.concatenate(features, axis=-1)


class GravityReconstructionNetwork(keras.Model):
    """
    Gravity Field Reconstruction Network

    Architecture:
    - 1 initial feature extraction layer
    - 6 dense blocks (each with 6 conv layers) = 36 conv layers total
    - 1 global feature fusion layer (1x1 conv)
    - 1 reconstruction + 1 deconvolution layer for upscaling

    Input: 30x30 low-resolution gravity patches
    Output: 30x30 high-resolution gravity patches (coarse)
    """
    def __init__(self, growth_rate=16, num_blocks=6, **kwargs):
        super(GravityReconstructionNetwork, self).__init__(**kwargs)
        self.growth_rate = growth_rate
        self.num_blocks = num_blocks

        # initial feature extraction layer
        self.feature_extraction = layers.Conv2D(
            filters=64,  # initial number of feature maps
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name='initial_feature_extraction'
        )

        # 6 dense blocks (each with 6 conv layers)
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(
                DenseBlock(growth_rate=growth_rate, num_layers=6, name=f'dense_block_{i+1}')
            )

        # global feature fusion layer (1x1 convolution)
        # fuses features from all dense blocks
        self.global_fusion = layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name='global_feature_fusion'
        )

        self.reconstruction = layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name='reconstruction'
        )

        # deconvolution layer for upscaling
        self.deconvolution = layers.Conv2DTranspose(
            filters=1,  # output single channel (gravity field)
            kernel_size=(3, 3),
            padding='same',
            activation=None,  # linear activation 
            kernel_initializer='he_normal',
            name='deconvolution'
        )

    def call(self, inputs):
        """
        Forward pass through the gravity reconstruction network

        Args:
            inputs: Low-resolution gravity field patches (batch_size, 30, 30, 1)

        Returns:
            Coarse high-resolution gravity field (batch_size, 30, 30, 1)
        """
        # initial feature extraction
        F0 = self.feature_extraction(inputs)

        # store all dense block outputs for global fusion
        dense_outputs = [F0]

        # F_d = f_d(F_{d-1}) for d = 1 to 6 - Dense blocks
        x = F0
        for dense_block in self.dense_blocks:
            x = dense_block(x)
            dense_outputs.append(x)

        # global feature fusion
        concatenated = layers.concatenate(dense_outputs, axis=-1)
        fused_features = self.global_fusion(concatenated)

        # reconstruction
        reconstructed = self.reconstruction(fused_features)

        # deconvolution for final output
        output = self.deconvolution(reconstructed)

        return output


class DEMRefiningNetwork(keras.Model):
    """
    DEM Refining Network

    Architecture:
    - Concatenation layer (coarse gravity + high-res DEM)
    - 3-layer residual network
    - Skip connection from coarse gravity to final output

    Input: Coarse gravity (30x30) + High-res DEM (30x30)
    Output: Refined high-resolution gravity field (30x30)
    """
    def __init__(self, **kwargs):
        super(DEMRefiningNetwork, self).__init__(**kwargs)

        # first convolutional layer
        self.conv1 = layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name='dem_conv1'
        )

        # second convolutional layer
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            name='dem_conv2'
        )

        # final convolutional layer (no activation)
        self.conv3 = layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            activation=None,
            kernel_initializer='he_normal',
            name='dem_conv3'
        )

    def call(self, coarse_gravity, high_res_dem):
        """
        Forward pass through DEM refining network

        Args:
            coarse_gravity: Output from gravity reconstruction network (batch_size, 30, 30, 1)
            high_res_dem: High-resolution DEM data (batch_size, 30, 30, 1)

        Returns:
            Refined high-resolution gravity field (batch_size, 30, 30, 1)
        """
        concatenated = layers.concatenate([coarse_gravity, high_res_dem], axis=-1)
        x = self.conv1(concatenated)
        x = self.conv2(x)
        residual = self.conv3(x)
        output = layers.add([residual, coarse_gravity])

        return output


class MercuryGravityReconstructionModel(keras.Model):
    """
    Combines gravity reconstruction network and DEM refining network
    """
    def __init__(self, growth_rate=16, num_blocks=6, **kwargs):
        super(MercuryGravityReconstructionModel, self).__init__(**kwargs)

        self.gravity_network = GravityReconstructionNetwork(
            growth_rate=growth_rate,
            num_blocks=num_blocks
        )

        self.dem_network = DEMRefiningNetwork()

    def call(self, inputs):
        """
        Forward pass through complete model

        Args:
            inputs: Tuple of (low_res_gravity, high_res_dem)
                low_res_gravity: (batch_size, 30, 30, 1)
                high_res_dem: (batch_size, 30, 30, 1)

        Returns:
            Refined high-resolution gravity field (batch_size, 30, 30, 1)
        """
        low_res_gravity, high_res_dem = inputs

        # gravity reconstruction
        coarse_gravity = self.gravity_network(low_res_gravity)
        # DEM refinement
        refined_gravity = self.dem_network(coarse_gravity, high_res_dem)

        return refined_gravity


def create_model(patch_size=30, growth_rate=16, num_blocks=6):
    """
    Create the complete model

    Args:
        patch_size: Size of input patches (default 30x30)
        growth_rate: Growth rate k for dense blocks (default 16)
        num_blocks: Number of dense blocks (default 6)

    Returns:
        Compiled Keras model ready for training
    """
    low_res_gravity_input = keras.Input(
        shape=(patch_size, patch_size, 1),
        name='low_res_gravity'
    )
    high_res_dem_input = keras.Input(
        shape=(patch_size, patch_size, 1),
        name='high_res_dem'
    )

    # model
    model = MercuryGravityReconstructionModel(
        growth_rate=growth_rate,
        num_blocks=num_blocks
    )

    # outputs
    outputs = model([low_res_gravity_input, high_res_dem_input])

    # build functional model
    functional_model = keras.Model(
        inputs=[low_res_gravity_input, high_res_dem_input],
        outputs=outputs,
        name='mercury_gravity_reconstruction'
    )

    # compie with Adam optimizer and MAE loss 
    optimizer = keras.optimizers.Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    functional_model.compile(
        optimizer=optimizer,
        loss='mae',  
        metrics=['mae', 'mse']
    )

    return functional_model


def create_gravity_only_model(patch_size=30, growth_rate=16, num_blocks=6):
    """
    useful for pretraining without DEM data

    Args:
        patch_size: Size of input patches (default 30x30)
        growth_rate: Growth rate k for dense blocks (default 16)
        num_blocks: Number of dense blocks (default 6)

    Returns:
        Compiled Keras model (gravity reconstruction only)
    """
    # define input
    low_res_gravity_input = keras.Input(
        shape=(patch_size, patch_size, 1),
        name='low_res_gravity'
    )

    # create gravity network
    gravity_network = GravityReconstructionNetwork(
        growth_rate=growth_rate,
        num_blocks=num_blocks
    )

    # get outputs
    outputs = gravity_network(low_res_gravity_input)

    # build functional model
    functional_model = keras.Model(
        inputs=low_res_gravity_input,
        outputs=outputs,
        name='gravity_reconstruction_only'
    )

    # compile with Adam optimizer and MAE loss
    optimizer = keras.optimizers.Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    functional_model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae', 'mse']
    )

    return functional_model


if __name__ == "__main__":
    # test model creation
    print("Creating complete model...")
    model = create_model(patch_size=30)
    model.summary()

    print("\n" + "="*80 + "\n")
    print("Creating gravity-only model...")
    gravity_model = create_gravity_only_model(patch_size=30)
    gravity_model.summary()

    # test with dummy data
    print("\n" + "="*80 + "\n")
    print("Testing forward pass...")
    batch_size = 2
    dummy_low_res = np.random.randn(batch_size, 30, 30, 1).astype(np.float32)
    dummy_dem = np.random.randn(batch_size, 30, 30, 1).astype(np.float32)

    output = model.predict([dummy_low_res, dummy_dem], verbose=0)
    print(f"Input shape: {dummy_low_res.shape}")
    print(f"DEM shape: {dummy_dem.shape}")
    print(f"Output shape: {output.shape}")
    print("\nModel created successfully!")
