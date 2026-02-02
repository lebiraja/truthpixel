"""
Model Architecture Module for TruthPixel AI-Generated Image Detection.

This module defines the CNN architecture using EfficientNetB0 with Transfer Learning
for binary classification of real vs AI-generated images.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TruthPixelModel:
    """
    Model builder for AI-Generated Image Detection using EfficientNetB0.

    Attributes:
        input_shape: Input image dimensions
        learning_rate: Initial learning rate for optimizer
        l2_reg: L2 regularization factor
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        learning_rate: float = 0.001,
        l2_reg: float = 0.01
    ):
        """
        Initialize the model builder.

        Args:
            input_shape: Shape of input images (height, width, channels)
            learning_rate: Learning rate for Adam optimizer
            l2_reg: L2 regularization factor
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.model = None

    def build_model(
        self,
        freeze_base: bool = True,
        dropout_rate_1: float = 0.5,
        dropout_rate_2: float = 0.3
    ) -> tf.keras.Model:
        """
        Build the EfficientNetB0-based model for binary classification.

        Architecture:
            EfficientNetB0 (ImageNet weights)
            ↓
            GlobalAveragePooling2D
            ↓
            Dense(256, relu) + L2 + Dropout(0.5)
            ↓
            Dense(128, relu) + L2 + Dropout(0.3)
            ↓
            Dense(1, sigmoid)

        Args:
            freeze_base: Whether to freeze EfficientNetB0 base layers
            dropout_rate_1: Dropout rate for first dense layer
            dropout_rate_2: Dropout rate for second dense layer

        Returns:
            Compiled Keras model
        """
        logger.info("Building TruthPixel model...")

        # Load EfficientNetB0 with ImageNet weights
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling=None
        )

        # Freeze base model if specified
        base_model.trainable = not freeze_base

        if freeze_base:
            logger.info("✓ Base model frozen (transfer learning mode)")
        else:
            logger.info("✓ Base model unfrozen (fine-tuning mode)")

        # Build the model
        inputs = layers.Input(shape=self.input_shape, name='input_image')

        # Base model (let Keras handle training mode automatically)
        x = base_model(inputs)

        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pooling')(x)

        # First dense layer with regularization and dropout
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='dense_256'
        )(x)
        x = layers.Dropout(dropout_rate_1, name='dropout_1')(x)

        # Second dense layer with regularization and dropout
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name='dense_128'
        )(x)
        x = layers.Dropout(dropout_rate_2, name='dropout_2')(x)

        # Output layer (binary classification)
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            name='output'
        )(x)

        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='TruthPixel')

        logger.info("✓ Model architecture built successfully")

        return self.model

    def compile_model(
        self,
        model: Optional[tf.keras.Model] = None,
        learning_rate: Optional[float] = None
    ) -> tf.keras.Model:
        """
        Compile the model with optimizer, loss, and metrics.

        Args:
            model: Model to compile (uses self.model if None)
            learning_rate: Learning rate (uses self.learning_rate if None)

        Returns:
            Compiled Keras model
        """
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model to compile. Build model first.")

        if learning_rate is None:
            learning_rate = self.learning_rate

        logger.info(f"Compiling model with learning rate: {learning_rate}")

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]
        )

        logger.info("✓ Model compiled successfully")

        return model

    def unfreeze_base_layers(
        self,
        model: Optional[tf.keras.Model] = None,
        num_layers_to_unfreeze: int = 20
    ) -> tf.keras.Model:
        """
        Unfreeze the top N layers of the base model for fine-tuning.

        Args:
            model: Model to modify (uses self.model if None)
            num_layers_to_unfreeze: Number of layers to unfreeze from the top

        Returns:
            Modified model
        """
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model to modify. Build model first.")

        # Get the base model (EfficientNetB0)
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower():
                base_model = layer
                break

        if base_model is None:
            logger.warning("Could not find EfficientNetB0 base model")
            return model

        # Freeze all layers first
        base_model.trainable = True

        # Freeze all except the last num_layers_to_unfreeze layers
        total_layers = len(base_model.layers)
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

        logger.info(f"✓ Unfrozen top {num_layers_to_unfreeze} layers out of {total_layers} base layers")

        # Count trainable parameters
        trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

        logger.info(f"  Trainable params: {trainable_count:,}")
        logger.info(f"  Non-trainable params: {non_trainable_count:,}")

        return model

    def get_model_summary(
        self,
        model: Optional[tf.keras.Model] = None
    ) -> None:
        """
        Print model summary.

        Args:
            model: Model to summarize (uses self.model if None)
        """
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model to summarize. Build model first.")

        logger.info("\n" + "=" * 80)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 80)

        model.summary()

        # Additional info
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

        logger.info("\n" + "=" * 80)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
        logger.info("=" * 80 + "\n")

    def save_model(
        self,
        model: Optional[tf.keras.Model] = None,
        filepath: str = 'models/truthpixel_model.h5'
    ) -> None:
        """
        Save model to file.

        Args:
            model: Model to save (uses self.model if None)
            filepath: Path to save the model
        """
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model to save. Build model first.")

        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model.save(filepath)
        logger.info(f"✓ Model saved to {filepath}")

    def load_model(self, filepath: str) -> tf.keras.Model:
        """
        Load model from file.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded Keras model
        """
        logger.info(f"Loading model from {filepath}")

        self.model = tf.keras.models.load_model(filepath)

        logger.info("✓ Model loaded successfully")

        return self.model


def create_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 0.001,
    freeze_base: bool = True
) -> tf.keras.Model:
    """
    Convenience function to create and compile the model.

    Args:
        input_shape: Shape of input images
        learning_rate: Learning rate for optimizer
        freeze_base: Whether to freeze base model

    Returns:
        Compiled Keras model
    """
    model_builder = TruthPixelModel(
        input_shape=input_shape,
        learning_rate=learning_rate
    )

    model = model_builder.build_model(freeze_base=freeze_base)
    model = model_builder.compile_model(model, learning_rate=learning_rate)

    return model


def main():
    """
    Main function to test model creation.
    """
    logger.info("Testing TruthPixel model creation...")

    # Create model builder
    model_builder = TruthPixelModel(
        input_shape=(224, 224, 3),
        learning_rate=0.001,
        l2_reg=0.01
    )

    # Build model with frozen base
    model = model_builder.build_model(freeze_base=True)

    # Compile model
    model = model_builder.compile_model(model)

    # Display summary
    model_builder.get_model_summary(model)

    # Test unfreezing
    logger.info("\nTesting layer unfreezing...")
    model = model_builder.unfreeze_base_layers(model, num_layers_to_unfreeze=20)

    # Recompile with lower learning rate for fine-tuning
    model = model_builder.compile_model(model, learning_rate=0.0001)

    logger.info("\n✓ Model creation test completed successfully!")


if __name__ == "__main__":
    main()
