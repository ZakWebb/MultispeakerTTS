import tensorflow as tf
from tf.keras import layers


def conv3x3(filters, stride=1):
    """3x3 convolution with padding"""
    return layers.Conv2D(filters, 3, strides=stride, use_bias=False, padding="same")


class Sequential(layers.Layer):
    def __init__(self, layers=[], **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.layers = layers
        self.built = False

    def add(self, layer):
        if self.built:
            raise RuntimeError("You can't call add to Sequential when the model has already been built.")
        self.layers.append(layer)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class BasicResNetBlock(layers.Layer):
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None, **kwargs):
        super(BasicResNetBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3(filters, stride)
        self.bn1 = layers.BatchNormalization(axis=1)
        self.relu = layers.ReLU()
        self.conv2 = conv3x3(filters)
        self.bn2 = layers.BatchNormalization(axis=1)
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(layers.Layer):
    def __init__(self, block, layers, initial_filter_dim=16, **kwargs):
        super(ResNET, self).__init__(**kwargs)
        self.initial_filter_dim = initial_filter_dim
        self.conv1 = layers.Conv2D(initial_filter_dim, 7, use_bias=False, padding="same")
        self.bn1 = layers.BatchNormalization(axis=1)
        self.relu = layers.ReLU()

        self.layer1 = self._make_layer(block, initial_filter_dim, initial_filter_dim, layers[0])
        self.layer2 = self._make_layer(block, i2 * initial_filter_dim, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * initial_filter_dim, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * initial_filter_dim, layers[3], stride=2)

        self.avgpool = layers.AveragePooling2D((1, 3), padding="same")

    def _make_layer(self, block, prev_filters, filters, blocks, stride=1):
        downsample = None
        if stride != 1 or prev_filters != filters * block.expansion:
            downsample = Sequential([
                layers.Conv2D(planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                layers.BatchNormalization(axis=1)
            ])

        layers = []
        layers.append(block(filters, stride=stride, downsample))
        for i in range(1, blocks):
            layers.append(block(filters))

        return Sequential(layers)

    def call(self, x):
        x = tf.expand_dims(x, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = tf.squeeze(x, [3])
        x = tf.transpose(x, perm=[0, 2, 1])

        return x


def resnet18(**kwargs):
    model = ResNet(BasicResNetBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicResNetBlock, [3, 4, 6, 3], **kwargs)
    return model


def thin_resnet_34(**kwargs):
    model = ResNet(BasicResNetBlock, [3, 4, 6, 3], initial_filter_dim=8, **kwargs)
    return model
