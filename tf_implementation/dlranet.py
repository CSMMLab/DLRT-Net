'''
brief: Reference_model
Author: Steffen Schotthöfer
Version: 0.0
Date 12.04.2022
'''
import tensorflow as tf
from tensorflow import keras
from os import path, makedirs
import numpy as np


class DLRANetDenseOut(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="e2eDLRANet", tol=0.4, low_rank=20, dlra_layer_dim=200,
                 rmax_total=100, **kwargs):
        super(DLRANetDenseOut, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.dlraBlockInput = DLRALayer(input_dim=input_dim, units=dlra_layer_dim, low_rank=dlra_layer_dim,
                                        epsAdapt=tol,
                                        rmax_total=rmax_total, )
        self.dlraBlock1 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock2 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock3 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlockOutput = Linear(input_dim=dlra_layer_dim, units=output_dim)

    def call(self, inputs, step: int = 0):
        z = self.dlraBlockInput(inputs, step=step)
        z = self.dlraBlock1(z, step=step)
        z = self.dlraBlock2(z, step=step)
        z = self.dlraBlock3(z, step=step)
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


class END2ENDDLRANet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="e2eDLRANet", tol=0.4, low_rank=20, dlra_layer_dim=200,
                 rmax_total=100, **kwargs):
        super(END2ENDDLRANet, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.dlraBlockInput = DLRALayer(input_dim=input_dim, units=dlra_layer_dim, low_rank=dlra_layer_dim,
                                        epsAdapt=tol,
                                        rmax_total=rmax_total, )
        self.dlraBlock1 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock2 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock3 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlockOutput = DLRALayerOutput(input_dim=dlra_layer_dim, units=output_dim, low_rank=output_dim,
                                               epsAdapt=tol, rmax_total=rmax_total, )

    def call(self, inputs, step: int = 0):
        z = self.dlraBlockInput(inputs, step=step)
        z = self.dlraBlock1(z, step=step)
        z = self.dlraBlock2(z, step=step)
        z = self.dlraBlock3(z, step=step)
        z = self.dlraBlockOutput(z, step=step)
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
        # self.layers[-1].trainable = False  # Dense output

        return 0

    def toggle_s_step_training(self):
        # self.layers[0].trainable = True  # Dense input
        # self.layers[-1].trainable = True  # Dense output
        return 0

    def save(self, folder_name):
        self.dlraBlockInput.save(folder_name=folder_name, layer_id=0)
        self.dlraBlock1.save(folder_name=folder_name, layer_id=1)
        self.dlraBlock2.save(folder_name=folder_name, layer_id=2)
        self.dlraBlock3.save(folder_name=folder_name, layer_id=3)
        self.dlraBlockOutput.save(folder_name=folder_name, layer_id=4)
        return 0

    def load(self, folder_name):
        self.dlraBlockInput.load(folder_name=folder_name, layer_id=0)
        self.dlraBlock1.load(folder_name=folder_name, layer_id=1)
        self.dlraBlock2.load(folder_name=folder_name, layer_id=2)
        self.dlraBlock3.load(folder_name=folder_name, layer_id=3)
        self.dlraBlockOutput.load(folder_name=folder_name, layer_id=4)
        return 0


class FullDLRANet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="partDLRANet", tol=0.4, low_rank=20, dlra_layer_dim=200,
                 rmax_total=100, **kwargs):
        super(FullDLRANet, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.denseBlock = DenseBlock(units=dlra_layer_dim, input_dim=input_dim)
        self.dlraBlock1 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock2 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock3 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        # self.dlraBlock4 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
        #                            rmax_total=rmax_total, )

        self.outputBlock = Linear(input_dim=dlra_layer_dim, units=output_dim)

    def call(self, inputs, step: int = 0):
        z = self.denseBlock(inputs)
        z = self.dlraBlock1(z, step=step)
        z = self.dlraBlock2(z, step=step)
        z = self.dlraBlock3(z, step=step)
        z = self.outputBlock(z)
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
        self.layers[0].trainable = False  # Dense input
        self.layers[-1].trainable = False  # Dense output

        return 0

    def toggle_s_step_training(self):
        self.layers[0].trainable = True  # Dense input
        self.layers[-1].trainable = True  # Dense output
        return 0

    def save(self, folder_name):
        self.denseBlock.save(folder_name=folder_name)
        self.dlraBlock1.save(folder_name=folder_name, layer_id=1)
        self.dlraBlock2.save(folder_name=folder_name, layer_id=2)
        self.dlraBlock3.save(folder_name=folder_name, layer_id=3)
        self.outputBlock.save(folder_name=folder_name)
        return 0

    def load(self, folder_name):
        self.denseBlock.load(folder_name=folder_name)
        self.dlraBlock1.load(folder_name=folder_name, layer_id=1)
        self.dlraBlock2.load(folder_name=folder_name, layer_id=2)
        self.dlraBlock3.load(folder_name=folder_name, layer_id=3)
        self.outputBlock.load(folder_name=folder_name)
        return 0


class PartDLRANet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="partDLRANet", tol=0.4, low_rank=20, rmax_total=100, **kwargs):
        super(PartDLRANet, self).__init__(name=name, **kwargs)
        self.denseBlock = DenseBlock(units=250, input_dim=input_dim)
        self.dlraBlock = DLRALayer(input_dim=250, units=250, low_rank=low_rank, epsAdapt=tol, rmax_total=rmax_total, )
        self.outputBlock = DenseBlockOutputSmall(output_dim=output_dim)

    def call(self, inputs, step: int):
        z = self.denseBlock(inputs)
        z = self.dlraBlock(z, step=step)
        z = self.outputBlock(z)
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
        self.layers[0].trainable = False  # Dense input
        self.layers[2].trainable = False  # Dense output
        # self.dlraBlock.b.trainable = False
        # self.dlraBlock.s.trainable = False
        # self.dlraBlock.k.trainable = True
        # self.dlraBlock.l_t.trainable = True

        return 0

    def toggle_s_step_training(self):
        self.layers[0].trainable = True  # Dense input
        self.layers[2].trainable = True  # Dense output
        # self.dlraBlock.b.trainable = True
        # self.dlraBlock.s.trainable = True
        # self.dlraBlock.k.trainable = False
        # self.dlraBlock.l_t.trainable = False

        return 0


class DenseBlock(keras.layers.Layer):
    # self.layer1 = Linear(units=units)

    def __init__(self, units=32, input_dim=32):
        super(DenseBlock, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", name="w_", trainable=True)
        self.b = self.add_weight(shape=(units,), initializer="zeros", name="b_", trainable=True)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        z = tf.keras.activations.relu(z)
        return z

    def save(self, folder_name):
        w_np = self.w.numpy()
        np.save(folder_name + "/w_in.npy", w_np)
        b_np = self.b.numpy()
        np.save(folder_name + "/b_in.npy", b_np)
        return 0

    def load(self, folder_name):
        a_np = np.load(folder_name + "/w_in.npy")
        self.w = tf.Variable(initial_value=a_np,
                             trainable=True, name="w_", dtype=tf.float32)
        b_np = np.load(folder_name + "/b_in.npy")
        self.b1 = tf.Variable(initial_value=b_np,
                              trainable=True, name="b_", dtype=tf.float32)


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


class DenseBlockOutput(keras.layers.Layer):
    def __init__(self, output_dim=1, name="dense_output_block", **kwargs):
        super(DenseBlockOutput, self).__init__(name=name, **kwargs)
        self.layer3 = Linear(units=250)
        self.layer4 = Linear(units=output_dim)

    def call(self, inputs):
        z = tf.keras.activations.relu(inputs)
        z = self.layer3(z)
        z = tf.keras.activations.relu(z)
        z = self.layer4(z)
        return z


class DenseBlockOutputSmall(keras.layers.Layer):
    def __init__(self, output_dim=1, name="dense_output_block", **kwargs):
        super(DenseBlockOutputSmall, self).__init__(name=name, **kwargs)
        self.layer4 = Linear(units=output_dim)

    def call(self, inputs):
        z = self.layer4(inputs)
        return z


def eig_cutoff_tol_regularized(x):  # for regularization of eigenvaluue cuttoff tolerance
    return 1e-3 * tf.reduce_sum(tf.square(tf.math.reciprocal(x, name="_regularized_tol")))


class DLRALayer(keras.layers.Layer):
    def __init__(self, input_dim: int, units=32, low_rank=10, epsAdapt=0.1, rmax_total=100, name="dlra_block",
                 **kwargs):
        super(DLRALayer, self).__init__(**kwargs)
        self.epsAdapt = epsAdapt  # for unconventional integrator
        self.units = units
        self.low_rank = low_rank
        self.rmax_total = rmax_total
        self.input_dim = input_dim

        self.k = self.add_weight(shape=(input_dim, self.low_rank), initializer="random_normal",
                                 trainable=True, name="k_")
        self.l_t = self.add_weight(shape=(self.units, self.low_rank), initializer="random_normal",
                                   trainable=True, name="lt_")
        self.s = self.add_weight(shape=(self.low_rank, self.low_rank), initializer="random_normal",
                                 trainable=True, name="s_")
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True, name="b_")
        # auxiliary variables
        self.aux_U = self.add_weight(shape=(self.input_dim, self.low_rank), initializer="random_normal",
                                     trainable=False, name="aux_U")
        self.aux_Unp1 = self.add_weight(shape=(self.input_dim, self.low_rank), initializer="random_normal",
                                        trainable=False, name="aux_Unp1")
        self.aux_Vt = self.add_weight(shape=(self.low_rank, self.units), initializer="random_normal",
                                      trainable=False, name="Vt")
        self.aux_Vtnp1 = self.add_weight(shape=(self.low_rank, self.units), initializer="random_normal",
                                         trainable=False, name="vtnp1")
        self.aux_N = self.add_weight(shape=(self.low_rank, self.low_rank), initializer="random_normal",
                                     trainable=False, name="aux_N")
        self.aux_M = self.add_weight(shape=(self.low_rank, self.low_rank), initializer="random_normal",
                                     trainable=False, name="aux_M")
        # Todo: initializer with low rank

    # @tf.function
    def call(self, inputs, step: int = 0):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """
        if step == 0:  # k-step
            z = tf.matmul(tf.matmul(inputs, self.k), self.aux_Vt)
        elif step == 1:  # l-step
            z = tf.matmul(tf.matmul(inputs, self.aux_U), self.l_t)
        else:  # s-step
            z = tf.matmul(tf.matmul(tf.matmul(inputs, self.aux_Unp1), self.s), self.aux_Vtnp1)
        return tf.keras.activations.relu(z + self.b)

    def k_step_preprocessing(self, ):
        k = tf.matmul(self.aux_U, self.s)
        self.k = tf.Variable(initial_value=k, trainable=True, name="k_")
        return 0

    def k_step_postprocessing(self):
        aux_Unp1, _ = tf.linalg.qr(self.k)
        self.aux_Unp1 = tf.Variable(initial_value=aux_Unp1, trainable=False, name="aux_Unp1")
        self.aux_N = tf.matmul(tf.transpose(self.aux_Unp1), self.aux_U)
        return 0

    def k_step_postprocessing_adapt(self):
        aux_Unp1, _ = tf.linalg.qr(tf.concat((self.k, self.aux_U), axis=1))
        self.aux_Unp1 = tf.Variable(initial_value=aux_Unp1, trainable=False, name="aux_Unp1")
        self.aux_N = tf.matmul(tf.transpose(self.aux_Unp1), self.aux_U)
        return 0

    def l_step_preprocessing(self, ):
        l_t = tf.matmul(self.s, self.aux_Vt)
        self.l_t = tf.Variable(initial_value=l_t, trainable=True, name="lt_")
        return 0

    def l_step_postprocessing(self):
        aux_Vtnp1, _ = tf.linalg.qr(tf.transpose(self.l_t))
        self.aux_Vtnp1 = tf.transpose(aux_Vtnp1)
        self.aux_M = tf.matmul(self.aux_Vtnp1, tf.transpose(self.aux_Vt))
        return 0

    def l_step_postprocessing_adapt(self):
        aux_Vtnp1, _ = tf.linalg.qr(tf.concat((tf.transpose(self.l_t), tf.transpose(self.aux_Vt)), axis=1))
        self.aux_Vtnp1 = tf.transpose(aux_Vtnp1)
        self.aux_M = tf.matmul(self.aux_Vtnp1, tf.transpose(self.aux_Vt))
        return 0

    def s_step_preprocessing(self):
        self.aux_U = self.aux_Unp1
        self.aux_Vt = self.aux_Vtnp1
        s = tf.matmul(tf.matmul(self.aux_N, self.s), tf.transpose(self.aux_M))
        self.s = tf.Variable(initial_value=s, trainable=True, name="s_")
        return 0

    def rank_adaption(self):
        # 1) compute SVD of S
        d, u2, v2 = tf.linalg.svd(self.s)  # d=singular values, u2 = left singuar vecs, v2= right singular vecss
        # print(d.shape)
        tmp = 0.0
        tol = self.epsAdapt * tf.linalg.norm(d)  # absolute value treshold (try also relative one)
        rmax = int(tf.floor(d.shape[0] / 2))
        for j in range(0, 2 * rmax - 1):
            tmp = tf.linalg.norm(d[j:2 * rmax - 1])
            if tmp < tol:
                rmax = j
                break

        rmax = tf.minimum(rmax, self.rmax_total)
        rmax = tf.maximum(rmax, 2)

        # update s
        self.s = tf.linalg.tensor_diag(d[:rmax])
        # self.s = s

        # update u and v
        self.aux_U = tf.matmul(self.aux_U, u2[:, :rmax])
        self.aux_Vt = tf.matmul(v2[:rmax, :], self.aux_Vt)
        self.low_rank = rmax
        return 0

    def get_config(self):
        config = super(DLRALayer, self).get_config()
        config.update({"units": self.units})
        config.update({"low_rank": self.low_rank})
        return config

    def save(self, folder_name, layer_id):
        # main_variables
        k_np = self.k.numpy()
        np.save(folder_name + "/k" + str(layer_id) + ".npy", k_np)
        l_t_np = self.l_t.numpy()
        np.save(folder_name + "/l_t" + str(layer_id) + ".npy", l_t_np)
        s_np = self.s.numpy()
        np.save(folder_name + "/s" + str(layer_id) + ".npy", s_np)
        b_np = self.b.numpy()
        np.save(folder_name + "/b" + str(layer_id) + ".npy", b_np)
        # aux_variables
        aux_U_np = self.aux_U.numpy()
        np.save(folder_name + "/aux_U" + str(layer_id) + ".npy", aux_U_np)
        aux_Unp1_np = self.aux_Unp1.numpy()
        np.save(folder_name + "/aux_Unp1" + str(layer_id) + ".npy", aux_Unp1_np)
        aux_Vt_np = self.aux_Vt.numpy()
        np.save(folder_name + "/aux_Vt" + str(layer_id) + ".npy", aux_Vt_np)
        aux_Vtnp1_np = self.aux_Vtnp1.numpy()
        np.save(folder_name + "/aux_Vtnp1" + str(layer_id) + ".npy", aux_Vtnp1_np)
        aux_N_np = self.aux_N.numpy()
        np.save(folder_name + "/aux_N" + str(layer_id) + ".npy", aux_N_np)
        aux_M_np = self.aux_M.numpy()
        np.save(folder_name + "/aux_M" + str(layer_id) + ".npy", aux_M_np)
        return 0

    def load(self, folder_name, layer_id):

        # main variables
        k_np = np.load(folder_name + "/k" + str(layer_id) + ".npy")
        self.low_rank = k_np.shape[1]
        self.k = tf.Variable(initial_value=k_np,
                             trainable=True, name="k_", dtype=tf.float32)
        l_t_np = np.load(folder_name + "/l_t" + str(layer_id) + ".npy")
        self.l_t = tf.Variable(initial_value=l_t_np,
                               trainable=True, name="lt_", dtype=tf.float32)
        s_np = np.load(folder_name + "/s" + str(layer_id) + ".npy")
        self.s = tf.Variable(initial_value=s_np,
                             trainable=True, name="s_", dtype=tf.float32)
        # aux variables
        aux_U_np = np.load(folder_name + "/aux_U" + str(layer_id) + ".npy")
        self.aux_U = tf.Variable(initial_value=aux_U_np,
                                 trainable=True, name="aux_U", dtype=tf.float32)
        aux_Unp1_np = np.load(folder_name + "/aux_Unp1" + str(layer_id) + ".npy")
        self.aux_Unp1 = tf.Variable(initial_value=aux_Unp1_np,
                                    trainable=True, name="aux_Unp1", dtype=tf.float32)
        Vt_np = np.load(folder_name + "/aux_Vt" + str(layer_id) + ".npy")
        self.aux_Vt = tf.Variable(initial_value=Vt_np,
                                  trainable=True, name="Vt", dtype=tf.float32)
        vtnp1_np = np.load(folder_name + "/aux_Vtnp1" + str(layer_id) + ".npy")
        self.aux_Vtnp1 = tf.Variable(initial_value=vtnp1_np,
                                     trainable=True, name="vtnp1", dtype=tf.float32)
        aux_N_np = np.load(folder_name + "/aux_N" + str(layer_id) + ".npy")
        self.aux_N = tf.Variable(initial_value=aux_N_np,
                                 trainable=True, name="aux_N", dtype=tf.float32)
        aux_M_np = np.load(folder_name + "/aux_M" + str(layer_id) + ".npy")
        self.aux_M = tf.Variable(initial_value=aux_M_np,
                                 trainable=True, name="aux_M", dtype=tf.float32)
        return 0


class DLRALayerOutput(DLRALayer):
    def __init__(self, input_dim: int, units=32, low_rank=10, epsAdapt=0.1, rmax_total=100, name="dlra_block",
                 **kwargs):
        super(DLRALayerOutput, self).__init__(input_dim=input_dim, units=units, low_rank=low_rank, epsAdapt=epsAdapt,
                                              rmax_total=rmax_total, name="dlra_block_out", **kwargs)

    def call(self, inputs, step: int = 0):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """
        if step == 0:  # k-step
            z = tf.matmul(tf.matmul(inputs, self.k), self.aux_Vt)
        elif step == 1:  # l-step
            z = tf.matmul(tf.matmul(inputs, self.aux_U), self.l_t)
        else:  # s-step
            z = tf.matmul(tf.matmul(tf.matmul(inputs, self.aux_Unp1), self.s), self.aux_Vtnp1)
        return z + self.b


class ReferenceNet(keras.Model):

    def __init__(self, input_dim=10, output_dim=1, layer_dim=200, name="referenceNet", **kwargs):
        super(ReferenceNet, self).__init__(name=name, **kwargs)
        self.layer1 = Linear(units=layer_dim, input_dim=input_dim)
        self.layer2 = Linear(units=layer_dim, input_dim=layer_dim)
        self.layer3 = Linear(units=layer_dim, input_dim=layer_dim)
        self.layer4 = Linear(units=layer_dim, input_dim=layer_dim)
        self.layer5 = Linear(units=output_dim, input_dim=layer_dim)

    @tf.function
    def call(self, inputs):
        z = self.layer1(inputs)
        z = tf.keras.activations.relu(z)
        z = self.layer2(z)
        z = tf.keras.activations.relu(z)
        z = self.layer3(z)
        z = tf.keras.activations.relu(z)
        z = self.layer4(z)
        z = tf.keras.activations.relu(z)
        z = self.layer5(z)
        return z


# ------ utils below

def create_csv_logger_cb(folder_name: str):
    '''
    dynamically creates a csvlogger and tensorboard logger
    '''
    # check if dir exists
    if not path.exists(folder_name + '/historyLogs/'):
        makedirs(folder_name + '/historyLogs/')

    # checkfirst, if history file exists.
    logName = folder_name + '/historyLogs/history_001_'
    count = 1
    while path.isfile(logName + '.csv'):
        count += 1
        logName = folder_name + \
                  '/historyLogs/history_' + str(count).zfill(3) + '_'

    logFileName = logName + '.csv'
    # create logger callback
    f = open(logFileName, "a")

    return f, logFileName
