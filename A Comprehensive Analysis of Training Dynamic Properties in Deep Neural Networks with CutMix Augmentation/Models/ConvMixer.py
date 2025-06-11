import tensorflow as tf
from tensorflow.keras import layers, models

def activation_block(x, dropout_rate, training):
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout_rate)(x, training=training)
    return layers.BatchNormalization()(x, training=training)

def conv_stem(x, filters: int, patch_size: int, dropout_rate, training):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x, dropout_rate, training)

def conv_mixer_block(x, filters: int, kernel_size: int, dropout_rate, training):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Dropout(dropout_rate)(x, training=training)
    x = layers.Add()([activation_block(x, dropout_rate, training), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = layers.Dropout(dropout_rate)(x, training=training)
    x = activation_block(x, dropout_rate, training)

    return x

def get_conv_mixer_256_8(
    image_size=32, filters=512, depth=12, kernel_size=3, patch_size=2, num_classes=num_classes, dropout_rate=0.
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = layers.Input((image_size, image_size, 3))
#     x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size, dropout_rate, training=True)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size, dropout_rate, training=True)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dropout(dropout_rate)(x, training=True)  # Dropout with rate 0 before final dense layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)

model = get_conv_mixer_512_12(dropout_rate=0.0)