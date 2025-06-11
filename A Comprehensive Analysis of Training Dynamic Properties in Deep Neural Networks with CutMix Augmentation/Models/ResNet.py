import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import DepthwiseConv2D


class BottleNeck(tf.keras.Model):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, strides=1, dropout_rate=0.):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        # Depthwise convolution followed by pointwise convolution
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.pointwise_conv = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        # self.conv2 = layers.Conv2D(out_channels, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        
        self.conv3 = layers.Conv2D(self.expansion*out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(dropout_rate)
        
        if strides != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = tf.keras.Sequential([
                layers.Conv2D(self.expansion*out_channels, kernel_size=1, strides=strides, use_bias=False), 
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
            
    def call(self, x, training=True):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x), training=training))
        out = self.dropout1(out, training=training)
        
        
        # out = tf.keras.activations.relu(self.bn2(self.conv2(out), training=training))
        
        out = tf.keras.activations.relu(self.bn2(self.pointwise_conv(self.depthwise_conv(out)), training=training))

        out = self.dropout2(out, training=training)
        
        
        out = self.bn3(self.conv3(out), training=training)
        out = self.dropout3(out, training=training)
        
        
        out = layers.add([self.shortcut(x), out])
        out = tf.keras.activations.relu(out)
        return out

class BuildResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, dropout_rate=0.):
        super(BuildResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], strides=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], strides=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], strides=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], strides=2, dropout_rate=dropout_rate)
        self.avg_pool2d = layers.AveragePooling2D(pool_size=4)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, training=True):
        out = tf.keras.activations.relu(self.bn1(self.conv1(x), training=training))
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        out = self.avg_pool2d(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
    def _make_layer(self, block, out_channels, num_blocks, strides, dropout_rate):
        stride = [strides] + [1]*(num_blocks-1)
        layers_list = []
        for s in stride:
            layers_list.append(block(self.in_channels, out_channels, s, dropout_rate))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers_list)

def ResNet50(num_classes=num_classes, dropout_rate=dropout_rate):
    return BuildResNet(BottleNeck, [3, 4, 6, 3], num_classes,dropout_rate)
