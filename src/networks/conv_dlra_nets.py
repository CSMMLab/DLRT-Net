'''
brief: Reference_model
Author: Steffen Schotthöfer
Version: 0.0
Date 12.04.2022
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16

from networks.convolutional_layers import DLRALayerConvAdaptive, DLRALayerConv
from networks.dense_layers import DLRALayer, DLRALayerAdaptive


class VGG15DLRANHead(keras.Model):

    def __init__(self, low_rank=20, tol=0.07, rmax_total=100, image_dims=(32, 32, 3), output_dim=10,
                 name="VGG15DLRANHead",
                 **kwargs):
        super(VGG15DLRANHead, self).__init__(name=name, **kwargs)

        # vgg16_body
        self.vgg16_body = VGG16(weights='imagenet', include_top=False)
        # deactivate training for that
        self.vgg16_body.trainable = False

        # dlra_layer_dim = 250
        # self.dlraBlockInput = DLRALayerAdaptive(input_dim=input_dim, units=dlra_layer_dim, low_rank=low_rank,
        #                                        epsAdapt=tol,
        #                                        rmax_total=rmax_total, )
        self.flatten_layer = keras.layers.Flatten()

        self.dlraBlock1 = DLRALayerAdaptive(input_dim=512, units=4096, low_rank=low_rank,
                                            epsAdapt=tol,
                                            rmax_total=rmax_total, )
        self.dlraBlock2 = DLRALayerAdaptive(input_dim=4096, units=4096, low_rank=low_rank,
                                            epsAdapt=tol,
                                            rmax_total=rmax_total, )
        self.dlraBlockOutput = Linear(input_dim=4096, units=output_dim)

    def build_model(self):
        self.dlraBlock1.build_model()
        self.dlraBlock2.build_model()

        return 0

    def call(self, inputs, step: int = 0):
        z = self.vgg16_body(inputs)
        z = self.flatten_layer(z)
        z = self.dlraBlock1(z, step=step)
        z = self.dlraBlock2(z, step=step)
        z = self.dlraBlockOutput(z)

        return z

    def k_step_preprocessing(self):
        self.dlraBlock1.k_step_preprocessing()
        self.dlraBlock2.k_step_preprocessing()
        return 0

    def l_step_preprocessing(self):
        self.dlraBlock1.l_step_preprocessing()
        self.dlraBlock2.l_step_preprocessing()

    def k_step_postprocessing(self):
        self.dlraBlock1.k_step_postprocessing()
        self.dlraBlock2.k_step_postprocessing()
        return 0

    def l_step_postprocessing(self):
        self.dlraBlock1.l_step_postprocessing()
        self.dlraBlock2.l_step_postprocessing()
        return 0

    def k_step_postprocessing_adapt(self):
        self.dlraBlock1.k_step_postprocessing_adapt()
        self.dlraBlock2.k_step_postprocessing_adapt()
        return 0

    def l_step_postprocessing_adapt(self):
        self.dlraBlock1.l_step_postprocessing_adapt()
        self.dlraBlock2.l_step_postprocessing_adapt()
        return 0

    def rank_adaption(self):
        self.dlraBlock1.rank_adaption()
        self.dlraBlock2.rank_adaption()
        return 0

    def s_step_preprocessing(self):
        self.dlraBlock1.s_step_preprocessing()
        self.dlraBlock2.s_step_preprocessing()
        return 0

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
        ranks = [self.dlraBlock1.low_rank,
                 self.dlraBlock2.low_rank]
        return ranks


#  Convolutional DLRANets

class DLRANetConv(keras.Model):
    # VGG16 for Cifar10

    def __init__(self, low_rank=20, tol=0.07, rmax_total=100, image_dims=(32, 32, 3), output_dim=10,
                 name="DLRANetConv",
                 **kwargs):
        super(DLRANetConv, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.image_dims = image_dims
        self.output_dim = output_dim

        self.low_rank = low_rank
        self.tol = tol
        self.rmax_total = rmax_total

        # ---- architecture
        # block 1)
        self.dlraBlock1a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64,
                                         image_dims=image_dims)
        next_image_dims = self.dlraBlock1a.output_shape_conv
        self.dlraBlock1b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock1b.output_shape_conv
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 2)
        self.dlraBlock2a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=128,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2a.output_shape_conv
        self.dlraBlock2b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=128,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2b.output_shape_conv
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 3)
        self.dlraBlock3a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3a.output_shape_conv
        self.dlraBlock3b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3b.output_shape_conv
        self.pool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 4)
        self.dlraBlock4a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4a.output_shape_conv
        self.dlraBlock4b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4b.output_shape_conv
        self.pool4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 5)
        self.dlraBlock5a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock5a.output_shape_conv
        self.dlraBlock5b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
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
        self.dlraDense1 = DLRALayer(input_dim=next_layer_input[1], units=512, low_rank=self.low_rank,
                                    epsAdapt=self.tol, rmax_total=200)
        self.dlraDense2 = DLRALayer(input_dim=512, units=512, low_rank=self.low_rank,
                                    epsAdapt=self.tol, rmax_total=200)
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
        return 0

    # @tf.function
    def call(self, inputs, step: int = 0):
        z = self.dlraBlock1a(inputs, step=step)
        z = self.dlraBlock1b(z, step=step)
        z = self.pool1(z)
        z = self.dlraBlock2a(z, step=step)
        z = self.dlraBlock2b(z, step=step)
        z = self.pool2(z)
        z = self.dlraBlock3a(z, step=step)
        z = self.dlraBlock3b(z, step=step)
        z = self.pool3(z)
        z = self.dlraBlock4a(z, step=step)
        z = self.dlraBlock4b(z, step=step)
        z = self.pool4(z)
        z = self.dlraBlock5a(z, step=step)
        z = self.dlraBlock5b(z, step=step)
        z = self.pool5(z)
        z = self.flatten_layer(z)
        z = self.dlraDense1(z, step=step)
        z = self.dlraDense2(z, step=step)
        z = self.output_layer(z)
        return z

    @tf.function
    def k_step_preprocessing(self):
        self.dlraBlock1a.k_step_preprocessing()
        self.dlraBlock1b.k_step_preprocessing()
        self.dlraBlock2a.k_step_preprocessing()
        self.dlraBlock2b.k_step_preprocessing()
        self.dlraBlock3a.k_step_preprocessing()
        self.dlraBlock3b.k_step_preprocessing()
        self.dlraBlock4a.k_step_preprocessing()
        self.dlraBlock4b.k_step_preprocessing()
        self.dlraBlock5a.k_step_preprocessing()
        self.dlraBlock5b.k_step_preprocessing()
        self.dlraDense1.k_step_preprocessing()
        self.dlraDense2.k_step_preprocessing()
        return 0

    @tf.function
    def l_step_preprocessing(self):
        self.dlraBlock1a.l_step_preprocessing()
        self.dlraBlock1b.l_step_preprocessing()
        self.dlraBlock2a.l_step_preprocessing()
        self.dlraBlock2b.l_step_preprocessing()
        self.dlraBlock3a.l_step_preprocessing()
        self.dlraBlock3b.l_step_preprocessing()
        self.dlraBlock4a.l_step_preprocessing()
        self.dlraBlock4b.l_step_preprocessing()
        self.dlraBlock5a.l_step_preprocessing()
        self.dlraBlock5b.l_step_preprocessing()
        self.dlraDense1.l_step_preprocessing()
        self.dlraDense2.l_step_preprocessing()

    @tf.function
    def k_step_postprocessing(self):
        self.dlraBlock1a.k_step_postprocessing()
        self.dlraBlock1b.k_step_postprocessing()
        self.dlraBlock2a.k_step_postprocessing()
        self.dlraBlock2b.k_step_postprocessing()
        self.dlraBlock3a.k_step_postprocessing()
        self.dlraBlock3b.k_step_postprocessing()
        self.dlraBlock4a.k_step_postprocessing()
        self.dlraBlock4b.k_step_postprocessing()
        self.dlraBlock5a.k_step_postprocessing()
        self.dlraBlock5b.k_step_postprocessing()
        self.dlraDense1.k_step_postprocessing()
        self.dlraDense2.k_step_postprocessing()
        return 0

    @tf.function
    def l_step_postprocessing(self):
        self.dlraBlock1a.l_step_postprocessing()
        self.dlraBlock1b.l_step_postprocessing()
        self.dlraBlock2a.l_step_postprocessing()
        self.dlraBlock2b.l_step_postprocessing()
        self.dlraBlock3a.l_step_postprocessing()
        self.dlraBlock3b.l_step_postprocessing()
        self.dlraBlock4a.l_step_postprocessing()
        self.dlraBlock4b.l_step_postprocessing()
        self.dlraBlock5a.l_step_postprocessing()
        self.dlraBlock5b.l_step_postprocessing()
        self.dlraDense1.l_step_postprocessing()
        self.dlraDense2.l_step_postprocessing()
        return 0

    @tf.function
    def s_step_preprocessing(self):
        self.dlraBlock1a.s_step_preprocessing()
        self.dlraBlock1b.s_step_preprocessing()
        self.dlraBlock2a.s_step_preprocessing()
        self.dlraBlock2b.s_step_preprocessing()
        self.dlraBlock3a.s_step_preprocessing()
        self.dlraBlock3b.s_step_preprocessing()
        self.dlraBlock4a.s_step_preprocessing()
        self.dlraBlock4b.s_step_preprocessing()
        self.dlraBlock5a.s_step_preprocessing()
        self.dlraBlock5b.s_step_preprocessing()
        self.dlraDense1.s_step_preprocessing()
        self.dlraDense2.s_step_preprocessing()
        return 0

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
        ranks = [self.dlraBlock1a.low_rank,
                 self.dlraBlock1b.low_rank,
                 self.dlraBlock2a.low_rank,
                 self.dlraBlock2b.low_rank,
                 self.dlraBlock3a.low_rank,
                 self.dlraBlock3b.low_rank,
                 self.dlraBlock4a.low_rank,
                 self.dlraBlock4b.low_rank,
                 self.dlraBlock5a.low_rank,
                 self.dlraBlock5b.low_rank,
                 self.dlraDense1.low_rank,
                 self.dlraDense2.low_rank]
        return ranks


class DLRANetConvAdapt(keras.Model):
    # VGG16 for Cifar10

    def __init__(self, low_rank=20, tol=0.07, rmax_total=100, image_dims=(32, 32, 3), output_dim=10,
                 name="DLRANetConv",
                 **kwargs):
        super(DLRANetConvAdapt, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.image_dims = image_dims
        self.output_dim = output_dim

        self.low_rank = low_rank
        self.tol = tol
        self.rmax_total = rmax_total

        # ---- architecture
        # block 1)
        self.dlraBlock1a = DLRALayerConvAdaptive(low_rank=20, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64,
                                                 image_dims=image_dims)
        next_image_dims = self.dlraBlock1a.output_shape_conv
        self.dlraBlock1b = DLRALayerConvAdaptive(low_rank=20, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock1b.output_shape_conv
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 2)
        self.dlraBlock2a = DLRALayerConvAdaptive(low_rank=60, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=128,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2a.output_shape_conv
        self.dlraBlock2b = DLRALayerConvAdaptive(low_rank=60, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=128,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2b.output_shape_conv
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 3)
        self.dlraBlock3a = DLRALayerConvAdaptive(low_rank=100, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3a.output_shape_conv
        self.dlraBlock3b = DLRALayerConvAdaptive(low_rank=100, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3b.output_shape_conv
        self.pool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 4)
        self.dlraBlock4a = DLRALayerConvAdaptive(low_rank=200, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4a.output_shape_conv
        self.dlraBlock4b = DLRALayerConvAdaptive(low_rank=200, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4b.output_shape_conv
        self.pool4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 5)
        self.dlraBlock5a = DLRALayerConvAdaptive(low_rank=200, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                                 image_dims=next_image_dims)
        next_image_dims = self.dlraBlock5a.output_shape_conv
        self.dlraBlock5b = DLRALayerConvAdaptive(low_rank=200, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                                 stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
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
        self.dlraDense1 = DLRALayerAdaptive(input_dim=next_layer_input[1], units=512, low_rank=150,
                                            epsAdapt=self.tol, rmax_total=200)
        self.dlraDense2 = DLRALayerAdaptive(input_dim=512, units=512, low_rank=150,
                                            epsAdapt=self.tol, rmax_total=200)
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
        return 0

    def call(self, inputs, step: int = 0):
        z = self.dlraBlock1a(inputs, step=step)
        z = self.dlraBlock1b(z, step=step)
        z = self.pool1(z)
        z = self.dlraBlock2a(z, step=step)
        z = self.dlraBlock2b(z, step=step)
        z = self.pool2(z)
        z = self.dlraBlock3a(z, step=step)
        z = self.dlraBlock3b(z, step=step)
        z = self.pool3(z)
        z = self.dlraBlock4a(z, step=step)
        z = self.dlraBlock4b(z, step=step)
        z = self.pool4(z)
        z = self.dlraBlock5a(z, step=step)
        z = self.dlraBlock5b(z, step=step)
        z = self.pool5(z)
        z = self.flatten_layer(z)
        z = self.dlraDense1(z, step=step)
        z = self.dlraDense2(z, step=step)
        z = self.output_layer(z)
        return z

    def k_step_preprocessing(self):
        self.dlraBlock1a.k_step_preprocessing()
        self.dlraBlock1b.k_step_preprocessing()
        self.dlraBlock2a.k_step_preprocessing()
        self.dlraBlock2b.k_step_preprocessing()
        self.dlraBlock3a.k_step_preprocessing()
        self.dlraBlock3b.k_step_preprocessing()
        self.dlraBlock4a.k_step_preprocessing()
        self.dlraBlock4b.k_step_preprocessing()
        self.dlraBlock5a.k_step_preprocessing()
        self.dlraBlock5b.k_step_preprocessing()
        self.dlraDense1.k_step_preprocessing()
        self.dlraDense2.k_step_preprocessing()
        return 0

    def l_step_preprocessing(self):
        self.dlraBlock1a.l_step_preprocessing()
        self.dlraBlock1b.l_step_preprocessing()
        self.dlraBlock2a.l_step_preprocessing()
        self.dlraBlock2b.l_step_preprocessing()
        self.dlraBlock3a.l_step_preprocessing()
        self.dlraBlock3b.l_step_preprocessing()
        self.dlraBlock4a.l_step_preprocessing()
        self.dlraBlock4b.l_step_preprocessing()
        self.dlraBlock5a.l_step_preprocessing()
        self.dlraBlock5b.l_step_preprocessing()
        self.dlraDense1.l_step_preprocessing()
        self.dlraDense2.l_step_preprocessing()

    def k_step_postprocessing(self):
        self.dlraBlock1a.k_step_postprocessing()
        self.dlraBlock1b.k_step_postprocessing()
        self.dlraBlock2a.k_step_postprocessing()
        self.dlraBlock2b.k_step_postprocessing()
        self.dlraBlock3a.k_step_postprocessing()
        self.dlraBlock3b.k_step_postprocessing()
        self.dlraBlock4a.k_step_postprocessing()
        self.dlraBlock4b.k_step_postprocessing()
        self.dlraBlock5a.k_step_postprocessing()
        self.dlraBlock5b.k_step_postprocessing()
        self.dlraDense1.k_step_postprocessing()
        self.dlraDense2.k_step_postprocessing()
        return 0

    def l_step_postprocessing(self):
        self.dlraBlock1a.l_step_postprocessing()
        self.dlraBlock1b.l_step_postprocessing()
        self.dlraBlock2a.l_step_postprocessing()
        self.dlraBlock2b.l_step_postprocessing()
        self.dlraBlock3a.l_step_postprocessing()
        self.dlraBlock3b.l_step_postprocessing()
        self.dlraBlock4a.l_step_postprocessing()
        self.dlraBlock4b.l_step_postprocessing()
        self.dlraBlock5a.l_step_postprocessing()
        self.dlraBlock5b.l_step_postprocessing()
        self.dlraDense1.l_step_postprocessing()
        self.dlraDense2.l_step_postprocessing()
        return 0

    def k_step_postprocessing_adapt(self):
        self.dlraBlock1a.k_step_postprocessing_adapt()
        self.dlraBlock1b.k_step_postprocessing_adapt()
        self.dlraBlock2a.k_step_postprocessing_adapt()
        self.dlraBlock2b.k_step_postprocessing_adapt()
        self.dlraBlock3a.k_step_postprocessing_adapt()
        self.dlraBlock3b.k_step_postprocessing_adapt()
        self.dlraBlock4a.k_step_postprocessing_adapt()
        self.dlraBlock4b.k_step_postprocessing_adapt()
        self.dlraBlock5a.k_step_postprocessing_adapt()
        self.dlraBlock5b.k_step_postprocessing_adapt()
        self.dlraDense1.k_step_postprocessing_adapt()
        self.dlraDense2.k_step_postprocessing_adapt()
        return 0

    def l_step_postprocessing_adapt(self):
        self.dlraBlock1a.l_step_postprocessing_adapt()
        self.dlraBlock1b.l_step_postprocessing_adapt()
        self.dlraBlock2a.l_step_postprocessing_adapt()
        self.dlraBlock2b.l_step_postprocessing_adapt()
        self.dlraBlock3a.l_step_postprocessing_adapt()
        self.dlraBlock3b.l_step_postprocessing_adapt()
        self.dlraBlock4a.l_step_postprocessing_adapt()
        self.dlraBlock4b.l_step_postprocessing_adapt()
        self.dlraBlock5a.l_step_postprocessing_adapt()
        self.dlraBlock5b.l_step_postprocessing_adapt()
        self.dlraDense1.l_step_postprocessing_adapt()
        self.dlraDense2.l_step_postprocessing_adapt()
        return 0

    def rank_adaption(self):
        self.dlraBlock1a.rank_adaption()
        self.dlraBlock1b.rank_adaption()
        self.dlraBlock2a.rank_adaption()
        self.dlraBlock2b.rank_adaption()
        self.dlraBlock3a.rank_adaption()
        self.dlraBlock3b.rank_adaption()
        self.dlraBlock4a.rank_adaption()
        self.dlraBlock4b.rank_adaption()
        self.dlraBlock5a.rank_adaption()
        self.dlraBlock5b.rank_adaption()
        self.dlraDense1.rank_adaption()
        self.dlraDense2.rank_adaption()
        return 0

    def s_step_preprocessing(self):
        self.dlraBlock1a.s_step_preprocessing()
        self.dlraBlock1b.s_step_preprocessing()
        self.dlraBlock2a.s_step_preprocessing()
        self.dlraBlock2b.s_step_preprocessing()
        self.dlraBlock3a.s_step_preprocessing()
        self.dlraBlock3b.s_step_preprocessing()
        self.dlraBlock4a.s_step_preprocessing()
        self.dlraBlock4b.s_step_preprocessing()
        self.dlraBlock5a.s_step_preprocessing()
        self.dlraBlock5b.s_step_preprocessing()
        self.dlraDense1.s_step_preprocessing()
        self.dlraDense2.s_step_preprocessing()
        return 0

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
        ranks = [self.dlraBlock1a.low_rank,
                 self.dlraBlock1b.low_rank,
                 self.dlraBlock2a.low_rank,
                 self.dlraBlock2b.low_rank,
                 self.dlraBlock3a.low_rank,
                 self.dlraBlock3b.low_rank,
                 self.dlraBlock4a.low_rank,
                 self.dlraBlock4b.low_rank,
                 self.dlraBlock5a.low_rank,
                 self.dlraBlock5b.low_rank,
                 self.dlraDense1.low_rank,
                 self.dlraDense2.low_rank]
        return ranks


class DLRANetVGG16(keras.Model):

    def __init__(self, low_rank=20, tol=0.07, rmax_total=100, image_dims=(28, 28, 1), output_dim=10,
                 name="DLRANetConv",
                 **kwargs):
        super(DLRANetVGG16, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.image_dims = image_dims
        self.output_dim = output_dim

        self.low_rank = low_rank
        self.tol = tol
        self.rmax_total = rmax_total

        # ---- architecture
        # block 1)
        self.dlraBlock1a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64,
                                         image_dims=image_dims)
        next_image_dims = self.dlraBlock1a.output_shape_conv
        self.dlraBlock1b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock1b.output_shape_conv
        self.pool1 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 2)
        self.dlraBlock2a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=128,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2a.output_shape_conv
        self.dlraBlock2b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=64,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock2b.output_shape_conv
        self.pool2 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 3)
        self.dlraBlock3a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3a.output_shape_conv
        self.dlraBlock3b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=256,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock3b.output_shape_conv
        self.pool3 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 4)
        self.dlraBlock4a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4a.output_shape_conv
        self.dlraBlock4b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock4b.output_shape_conv
        self.pool4 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        # test for next image shapes
        t = tf.ones(shape=(1, next_image_dims[0], next_image_dims[1], next_image_dims[2]))
        out = self.pool1(t)
        next_image_dims = out.shape[1:]

        # block 5)
        self.dlraBlock5a = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
                                         image_dims=next_image_dims)
        next_image_dims = self.dlraBlock5a.output_shape_conv
        self.dlraBlock5b = DLRALayerConv(low_rank=self.low_rank, epsAdapt=self.tol, rmax_total=self.rmax_total,
                                         stride=(1, 1), rate=(1, 1), size=(3, 3), filters=512,
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
        self.dlraDense1 = DLRALayer(input_dim=next_layer_input[1], units=4096, low_rank=self.low_rank,
                                    epsAdapt=self.tol, rmax_total=200, )
        self.dlraDense2 = DLRALayer(input_dim=4096, units=4096, low_rank=self.low_rank,
                                    epsAdapt=self.tol, rmax_total=200, )
        self.outputLinear = Linear(input_dim=4096, units=self.output_dim)

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
        return 0

    # @tf.function
    def call(self, inputs, step: int = 0):
        z = self.dlraBlock1a(inputs, step=step)
        z = self.dlraBlock1b(z, step=step)
        z = self.pool1(z)
        z = self.dlraBlock2a(z, step=step)
        z = self.dlraBlock2b(z, step=step)
        z = self.pool2(z)
        z = self.dlraBlock3a(z, step=step)
        z = self.dlraBlock3b(z, step=step)
        z = self.pool3(z)
        z = self.dlraBlock4a(z, step=step)
        z = self.dlraBlock4b(z, step=step)
        z = self.pool4(z)
        z = self.dlraBlock5a(z, step=step)
        z = self.dlraBlock5b(z, step=step)
        z = self.pool5(z)
        z = self.flatten_layer(z)
        z = self.dlraDense1(z, step=step)
        z = self.dlraDense2(z, step=step)
        z = self.outputLinear(z)
        return z

    def k_step_preprocessing(self):
        self.dlraBlock1a.k_step_preprocessing()
        self.dlraBlock1b.k_step_preprocessing()
        self.dlraBlock2a.k_step_preprocessing()
        self.dlraBlock2b.k_step_preprocessing()
        self.dlraBlock3a.k_step_preprocessing()
        self.dlraBlock3b.k_step_preprocessing()
        self.dlraBlock4a.k_step_preprocessing()
        self.dlraBlock4b.k_step_preprocessing()
        self.dlraBlock5a.k_step_preprocessing()
        self.dlraBlock5b.k_step_preprocessing()
        self.dlraDense1.k_step_preprocessing()
        self.dlraDense2.k_step_preprocessing()
        return 0

    def l_step_preprocessing(self):
        self.dlraBlock1a.l_step_preprocessing()
        self.dlraBlock1b.l_step_preprocessing()
        self.dlraBlock2a.l_step_preprocessing()
        self.dlraBlock2b.l_step_preprocessing()
        self.dlraBlock3a.l_step_preprocessing()
        self.dlraBlock3b.l_step_preprocessing()
        self.dlraBlock4a.l_step_preprocessing()
        self.dlraBlock4b.l_step_preprocessing()
        self.dlraBlock5a.l_step_preprocessing()
        self.dlraBlock5b.l_step_preprocessing()
        self.dlraDense1.l_step_preprocessing()
        self.dlraDense2.l_step_preprocessing()

    def k_step_postprocessing(self):
        self.dlraBlock1a.k_step_postprocessing()
        self.dlraBlock1b.k_step_postprocessing()
        self.dlraBlock2a.k_step_postprocessing()
        self.dlraBlock2b.k_step_postprocessing()
        self.dlraBlock3a.k_step_postprocessing()
        self.dlraBlock3b.k_step_postprocessing()
        self.dlraBlock4a.k_step_postprocessing()
        self.dlraBlock4b.k_step_postprocessing()
        self.dlraBlock5a.k_step_postprocessing()
        self.dlraBlock5b.k_step_postprocessing()
        self.dlraDense1.k_step_postprocessing()
        self.dlraDense2.k_step_postprocessing()
        return 0

    def l_step_postprocessing(self):
        self.dlraBlock1a.l_step_postprocessing()
        self.dlraBlock1b.l_step_postprocessing()
        self.dlraBlock2a.l_step_postprocessing()
        self.dlraBlock2b.l_step_postprocessing()
        self.dlraBlock3a.l_step_postprocessing()
        self.dlraBlock3b.l_step_postprocessing()
        self.dlraBlock4a.l_step_postprocessing()
        self.dlraBlock4b.l_step_postprocessing()
        self.dlraBlock5a.l_step_postprocessing()
        self.dlraBlock5b.l_step_postprocessing()
        self.dlraDense1.l_step_postprocessing()
        self.dlraDense2.l_step_postprocessing()
        return 0

    def k_step_postprocessing_adapt(self):
        self.dlraBlock1a.k_step_postprocessing_adapt()
        self.dlraBlock1b.k_step_postprocessing_adapt()
        self.dlraBlock2a.k_step_postprocessing_adapt()
        self.dlraBlock2b.k_step_postprocessing_adapt()
        self.dlraBlock3a.k_step_postprocessing_adapt()
        self.dlraBlock3b.k_step_postprocessing_adapt()
        self.dlraBlock4a.k_step_postprocessing_adapt()
        self.dlraBlock4b.k_step_postprocessing_adapt()
        self.dlraBlock5a.k_step_postprocessing_adapt()
        self.dlraBlock5b.k_step_postprocessing_adapt()
        self.dlraDense1.k_step_postprocessing_adapt()
        self.dlraDense2.k_step_postprocessing_adapt()
        return 0

    def l_step_postprocessing_adapt(self):
        self.dlraBlock1a.l_step_postprocessing_adapt()
        self.dlraBlock1b.l_step_postprocessing_adapt()
        self.dlraBlock2a.l_step_postprocessing_adapt()
        self.dlraBlock2b.l_step_postprocessing_adapt()
        self.dlraBlock3a.l_step_postprocessing_adapt()
        self.dlraBlock3b.l_step_postprocessing_adapt()
        self.dlraBlock4a.l_step_postprocessing_adapt()
        self.dlraBlock4b.l_step_postprocessing_adapt()
        self.dlraBlock5a.l_step_postprocessing_adapt()
        self.dlraBlock5b.l_step_postprocessing_adapt()
        self.dlraDense1.l_step_postprocessing_adapt()
        self.dlraDense2.l_step_postprocessing_adapt()
        return 0

    def rank_adaption(self):
        self.dlraBlock1a.rank_adaption()
        self.dlraBlock1b.rank_adaption()
        self.dlraBlock2a.rank_adaption()
        self.dlraBlock2b.rank_adaption()
        self.dlraBlock3a.rank_adaption()
        self.dlraBlock3b.rank_adaption()
        self.dlraBlock4a.rank_adaption()
        self.dlraBlock4b.rank_adaption()
        self.dlraBlock5a.rank_adaption()
        self.dlraBlock5b.rank_adaption()
        self.dlraDense1.rank_adaption()
        self.dlraDense2.rank_adaption()
        return 0

    def s_step_preprocessing(self):
        self.dlraBlock1a.s_step_preprocessing()
        self.dlraBlock1b.s_step_preprocessing()
        self.dlraBlock2a.s_step_preprocessing()
        self.dlraBlock2b.s_step_preprocessing()
        self.dlraBlock3a.s_step_preprocessing()
        self.dlraBlock3b.s_step_preprocessing()
        self.dlraBlock4a.s_step_preprocessing()
        self.dlraBlock4b.s_step_preprocessing()
        self.dlraBlock5a.s_step_preprocessing()
        self.dlraBlock5b.s_step_preprocessing()
        self.dlraDense1.s_step_preprocessing()
        self.dlraDense2.s_step_preprocessing()
        return 0

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
        self.dlraBlockInput.save(folder_name=folder_name, layer_id=0)
        self.dlraBlock1.save(folder_name=folder_name, layer_id=1)
        self.dlraBlock2.save(folder_name=folder_name, layer_id=2)
        self.dlraBlock3.save(folder_name=folder_name, layer_id=3)
        self.dlraBlockOutput.save(folder_name=folder_name)
        return 0

    def load(self, folder_name):
        self.dlraBlockInput.load(folder_name=folder_name, layer_id=0)
        self.dlraBlock1.load(folder_name=folder_name, layer_id=1)
        self.dlraBlock2.load(folder_name=folder_name, layer_id=2)
        self.dlraBlock3.load(folder_name=folder_name, layer_id=3)
        self.dlraBlockOutput.load(folder_name=folder_name)
        return 0


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, name="linear", **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal",
                                 trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config

    def save(self, folder_name):
        w_np = self.w.numpy()
        np.save(folder_name + "/w_out.npy", w_np)
        b_np = self.b.numpy()
        np.save(folder_name + "/b_out.npy", b_np)
        return 0

    def load(self, folder_name):
        a_np = np.load(folder_name + "/w_out.npy")
        self.w = tf.Variable(initial_value=a_np,
                             trainable=True, name="w_", dtype=tf.float32)
        b_np = np.load(folder_name + "/b_out.npy")
        self.b1 = tf.Variable(initial_value=b_np,
                              trainable=True, name="b_", dtype=tf.float32)
