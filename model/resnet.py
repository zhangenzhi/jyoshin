import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_array_ops import identity


class ResNet(keras.Model):
    def __init__(self,
                 model_type="ResNet50",
                 include_top=True,
                 preact=False,
                 use_bias=True,
                 pooling=None,
                 classes=1000,
                 classifier_activation='softmax',
                 ):
        super(ResNet, self).__init__()

        self.model_type = model_type
        self.include_top = include_top
        self.preact = preact
        self.use_bias = use_bias
        self.pooling = pooling
        self.classes = classes
        self.classifier_activation = classifier_activation

        self.head = self._build_head()
        self.stem = self._build_stack_fn()
        self.top = self._build_top()

    def _build_head(self):
        head = []
        bn_axis = 3

        head.append(layers.ZeroPadding2D(
            padding=((3, 3), (3, 3)), name='conv1_pad'))
        head.append(layers.Conv2D(64, 7, strides=2,
                                  use_bias=self.use_bias, name='conv1_conv'))

        if not self.preact:
            head.append(layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='conv1_bn'))
            head.append(layers.Activation('relu', name='conv1_relu'))

    def _build_top(self):
        top = []
        bn_axis = 3

        if self.preact:
            top.append(layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='post_bn'))
            top.append(layers.Activation('relu', name='post_relu'))

        if self.include_top:
            top.append(layers.GlobalAveragePooling2D(name='avg_pool'))
            top.append(layers.Dense(self.classes, activation=self.classifier_activation,
                                    name='predictions'))
        else:
            if self.pooling == 'avg':
                top.append(layers.GlobalAveragePooling2D(name='avg_pool'))
            elif self.pooling == 'max':
                top.append(layers.GlobalMaxPooling2D(name='max_pool'))
        return top

    def _build_block(filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):

        bn_axis = 3
        layer = []
        shortcut = []

        if conv_shortcut:
            shortcut.append(layers.Conv2D(
                4 * filters, 1, strides=stride, name=name + '_0_conv'))
            shortcut.append(layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn'))
        else:
            shortcut.append(layers.Lambda(lambda x: x))

        layer.append(layers.Conv2D(
            filters, 1, strides=stride, name=name + '_1_conv'))
        layer.append(layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn'))
        layer.append(layers.Activation('relu', name=name + '_1_relu'))

        layer.append(layers.Conv2D(
            filters, kernel_size, padding='SAME', name=name + '_2_conv'))
        layer.append(layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn'))
        layer.append(layers.Activation('relu', name=name + '_2_relu'))

        layer.append(layers.Conv2D(4 * filters, 1, name=name + '_3_conv'))
        layer.append(layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn'))

        layer.append(layers.Add(name=name + '_add')) # [shortcut,x]
        layer.append(layers.Activation('relu', name=name + '_out'))
        return layer, shortcut

    def _build_stack(self, filters, blocks, stride1=2, name=None):
        stack = []
        stack.append(self._build_block(
            filters, stride=stride1, name=name + '_block1'))
        for i in range(2, blocks + 1):
            stack.append(self._build_block(filters, conv_shortcut=False,
                         name=name + '_block' + str(i)))
        return stack

    def _build_stack_fn(self):
        stem = []
        if self.model_type == "ResNet50":
            stem.append(self._build_stack(
                filters=64, blocks=3, stride1=1, name='conv2'))
            stem.append(self._build_stack(
                filters=128, blocks=4, stride1=2, name='conv3'))
            stem.append(self._build_stack(
                filters=256, blocks=6, stride1=2, name='conv4'))
            stem.append(self._build_stack(
                filters=512, blocks=3, stride1=2, name='conv5'))
        else:
            print("No such resnet.")
        return stem

    def call(self, inputs, training=None, mask=None):
        x = inputs
        
        # head
        for layer in self.head:
            x = layer(x)
            
        # stem
        for stack in self.stem:
            for block in stack:
                conv_layers, shortcuts = block
                for layer in conv_layers:
                    if isinstance(layer, tf.keras.layers.Add):
                        s = x
                        for shortcut in shortcuts:
                            s = shortcut(s)
                        x = layer([x,s])
                    else:
                        x = layer(x)
        
        # top
        for layer in self.top:
            x = layer(x)

        return x
