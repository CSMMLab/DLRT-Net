'''
brief: Reference_model
Author: Steffen Schotthöfer
Version: 0.0
Date 12.04.2022
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

from networks.convolutional_layers import LayerConv
from networks.dense_layers import DLRALayer, DLRALayerAdaptive, Linear, DenseLinear
from tensorflow.keras.applications.vgg16 import VGG16


#  Convolutional Networks (no DLRA)
class VGG15DLRANHead_NoDLRA(keras.Model):

    def __init__(self, low_rank=20, tol=0.07, rmax_total=100, image_dims=(32, 32, 3), output_dim=10,
                 name="VGG15DLRANHead",
                 **kwargs):
        super(VGG15DLRANHead_NoDLRA, self).__init__(name=name, **kwargs)

        # vgg16_body
        self.vgg16_body = VGG16(weights='imagenet', include_top=False)
        # deactivate training for that
        self.vgg16_body.trainable = True

        self.flatten_layer = keras.layers.Flatten()
        self.dlraBlock1 = DenseLinear(input_dim=512, units=4096)
        self.dlraBlock2 = DenseLinear(input_dim=4096, units=4096)
        self.dlraBlockOutput = Linear(input_dim=4096, units=output_dim)

    def build_model(self):
        self.dlraBlock1.build_model()
        self.dlraBlock2.build_model()
        self.dlraBlockOutput.build_model()
        return 0

    def call(self, inputs, step: int = 0):
        z = self.vgg16_body(inputs)
        z = self.flatten_layer(z)
        z = self.dlraBlock1(z)
        z = self.dlraBlock2(z)
        z = self.dlraBlockOutput(z)

        return z

    @staticmethod
    def set_none_grads_to_zero(grads, weights):
        """
        :param grads: gradients of current tape
        :param weights: weights of current model
        :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
        """
        for i in range(len(grads)):
            if grads[i] is None:
                grads[i] = tf.zeros(shape=weights[i].shape, dtype=tf.float32)
        return 0

    @staticmethod
    def set_dlra_bias_grads_to_zero(grads):
        """
        :param grads: gradients of current tape
        :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
        """
        for i in range(len(grads)):
            if len(grads[i].shape) == 1:
                grads[i] = tf.math.scalar_mul(0.0, grads[i])
        return 0

    def save(self, folder_name):
        self.dlraBlock1.save(folder_name=folder_name, layer_id=0)
        self.dlraBlock2.save(folder_name=folder_name, layer_id=1)
        self.dlraBlockOutput.save(folder_name=folder_name)

        return 0

    def load(self, folder_name):
        self.dlraBlock1.load(folder_name=folder_name, layer_id=0)
        self.dlraBlock2.load(folder_name=folder_name, layer_id=1)
        self.dlraBlockOutput.load(folder_name=folder_name)

        return 0

    def get_low_ranks(self):
        ranks = []
        return ranks


class VGG16Conv(keras.Model):
    # VGG16 for Cifar10

    def __init__(self, image_dims=(32, 32, 3), output_dim=10,
                 name="DLRANetConv",
                 **kwargs):
        super(VGG16Conv, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.image_dims = image_dims
        self.output_dim = output_dim

        # ---- architecture
        # block 1)
        self.dlraBlock1a = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64, image_dims=image_dims)
        next_image_dims = self.dlraBlock1a.output_shape_conv
        self.dlraBlock1b = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64, image_dims=next_image_dims)
        next_image_dims = self.dlraBlock1b.output_shape_conv
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 2)
        self.dlraBlock2a = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=128, image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2a.output_shape_conv
        self.dlraBlock2b = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=128,
                                     image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2b.output_shape_conv
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 3)
        self.dlraBlock3a = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                     image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3a.output_shape_conv
        self.dlraBlock3b = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                     image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3b.output_shape_conv
        self.pool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 4)
        self.dlraBlock4a = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                     image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4a.output_shape_conv
        self.dlraBlock4b = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                     image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4b.output_shape_conv
        self.pool4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 5)
        self.dlraBlock5a = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                     image_dims=next_image_dims)
        next_image_dims = self.dlraBlock5a.output_shape_conv
        self.dlraBlock5b = LayerConv(stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                     image_dims=next_image_dims)
        next_image_dims = self.dlraBlock5b.output_shape_conv
        self.pool5 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # dense blocks
        self.flatten_layer = keras.layers.Flatten()
        t = tf.ones(shape=(4, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.flatten_layer(t)
        next_layer_input = out.shape
        self.dlraDense1 = DenseLinear(input_dim=next_layer_input[1], units=512)
        self.dlraDense2 = DenseLinear(input_dim=512, units=512)
        self.output_layer = Linear(input_dim=512, units=self.output_dim)

    def build_model(self):
        self.dlraBlock1a.build_model()
        self.dlraBlock1b.build_model()
        self.dlraBlock2a.build_model()
        self.dlraBlock2b.build_model()
        self.dlraBlock3a.build_model()
        self.dlraBlock3b.build_model()
        self.dlraBlock4a.build_model()
        self.dlraBlock4b.build_model()
        self.dlraBlock5a.build_model()
        self.dlraBlock5b.build_model()

        self.dlraDense1.build_model()
        self.dlraDense2.build_model()
        self.output_layer.build_model()

        return 0

    # @tf.function
    def call(self, inputs):
        z = self.dlraBlock1a(inputs)
        z = self.dlraBlock1b(z)
        z = self.pool1(z)
        z = self.dlraBlock2a(z)
        z = self.dlraBlock2b(z)
        z = self.pool2(z)
        z = self.dlraBlock3a(z)
        z = self.dlraBlock3b(z)
        z = self.pool3(z)
        z = self.dlraBlock4a(z)
        z = self.dlraBlock4b(z)
        z = self.pool4(z)
        z = self.dlraBlock5a(z)
        z = self.dlraBlock5b(z)
        z = self.pool5(z)
        z = self.flatten_layer(z)
        z = self.dlraDense1(z)
        z = self.dlraDense2(z)
        z = self.output_layer(z)
        return z

    @staticmethod
    def set_none_grads_to_zero(grads, weights):
        """
        :param grads: gradients of current tape
        :param weights: weights of current model
        :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
        """
        for i in range(len(grads)):
            if grads[i] is None:
                grads[i] = tf.zeros(shape=weights[i].shape, dtype=tf.float32)
        return 0

    @staticmethod
    def set_dlra_bias_grads_to_zero(grads):
        """
        :param grads: gradients of current tape
        :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
        """
        for i in range(len(grads)):
            if len(grads[i].shape) == 1:
                grads[i] = tf.math.scalar_mul(0.0, grads[i])
        return 0

    def toggle_non_s_step_training(self):
        # self.layers[0].trainable = False  # Dense input
        self.layers[-1].trainable = False  # Dense output

        return 0

    def toggle_s_step_training(self):
        # self.layers[0].trainable = True  # Dense input
        self.layers[-1].trainable = True  # Dense output
        return 0

    def save(self, folder_name):
        self.dlraBlock1a.save(folder_name=folder_name, layer_id=0)
        self.dlraBlock1b.save(folder_name=folder_name, layer_id=1)
        self.dlraBlock2a.save(folder_name=folder_name, layer_id=2)
        self.dlraBlock2b.save(folder_name=folder_name, layer_id=3)
        self.dlraBlock3a.save(folder_name=folder_name, layer_id=4)
        self.dlraBlock3b.save(folder_name=folder_name, layer_id=5)
        self.dlraBlock4a.save(folder_name=folder_name, layer_id=6)
        self.dlraBlock4b.save(folder_name=folder_name, layer_id=7)
        self.dlraBlock5a.save(folder_name=folder_name, layer_id=8)
        self.dlraBlock5b.save(folder_name=folder_name, layer_id=9)

        self.dlraDense1.save(folder_name=folder_name, layer_id=10)
        self.dlraDense2.save(folder_name=folder_name, layer_id=11)
        self.output_layer.save(folder_name=folder_name)

        return 0

    def load(self, folder_name):
        self.dlraBlock1a.load(folder_name=folder_name, layer_id=0)
        self.dlraBlock1b.load(folder_name=folder_name, layer_id=1)
        self.dlraBlock2a.load(folder_name=folder_name, layer_id=2)
        self.dlraBlock2b.load(folder_name=folder_name, layer_id=3)
        self.dlraBlock3a.load(folder_name=folder_name, layer_id=4)
        self.dlraBlock3b.load(folder_name=folder_name, layer_id=5)
        self.dlraBlock4a.load(folder_name=folder_name, layer_id=6)
        self.dlraBlock4b.load(folder_name=folder_name, layer_id=7)
        self.dlraBlock5a.load(folder_name=folder_name, layer_id=8)
        self.dlraBlock5b.load(folder_name=folder_name, layer_id=9)

        self.dlraDense1.load(folder_name=folder_name, layer_id=10)
        self.dlraDense2.load(folder_name=folder_name, layer_id=11)
        self.output_layer.load(folder_name=folder_name)

        return 0

    def get_low_ranks(self):
        ranks = []
        return ranks
