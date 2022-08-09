import tensorflow as tf
from tensorflow import keras
from os import path, makedirs
import numpy as np


class NaiveNet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="e2eDLRANet", tol=0.4, low_rank=20, dlra_layer_dim=200,
                 rmax_total=100, **kwargs):
        super(NaiveNet, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.input_dim = input_dim
        self.dlra_layer_dim = dlra_layer_dim
        self.low_rank = low_rank
        self.output_dim = output_dim
        self.tol = tol
        self.rmax_total = rmax_total

        self.dlraBlockInput = NaiveFactorLayer(input_dim=self.input_dim, units=self.dlra_layer_dim,
                                               low_rank=self.low_rank,
                                               epsAdapt=self.tol, rmax_total=self.rmax_total, )
        self.dlraBlock1 = NaiveFactorLayer(input_dim=self.dlra_layer_dim, units=self.dlra_layer_dim,
                                           low_rank=self.low_rank,
                                           epsAdapt=self.tol,
                                           rmax_total=self.rmax_total, )
        self.dlraBlock2 = NaiveFactorLayer(input_dim=self.dlra_layer_dim, units=self.dlra_layer_dim,
                                           low_rank=self.low_rank,
                                           epsAdapt=self.tol,
                                           rmax_total=self.rmax_total, )
        self.dlraBlock3 = NaiveFactorLayer(input_dim=self.dlra_layer_dim, units=self.dlra_layer_dim,
                                           low_rank=self.low_rank,
                                           epsAdapt=self.tol,
                                           rmax_total=rmax_total, )
        self.dlraBlockOutput = Linear2(input_dim=self.dlra_layer_dim, units=self.output_dim)

    def build_model(self):
        self.dlraBlockInput.build_model()
        self.dlraBlock1.build_model()
        self.dlraBlock2.build_model()
        self.dlraBlock3.build_model()
        return 0

    @tf.function
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
        self.dlraBlockOutput.save(folder_name=folder_name, layer_id=4)
        return 0

    def load(self, folder_name):
        self.dlraBlockInput.load(folder_name=folder_name, layer_id=0)
        self.dlraBlock1.load(folder_name=folder_name, layer_id=1)
        self.dlraBlock2.load(folder_name=folder_name, layer_id=2)
        self.dlraBlock3.load(folder_name=folder_name, layer_id=3)
        self.dlraBlockOutput.load(folder_name=folder_name, layer_id=4)
        return 0

    def load_from_fullW(self, folder_name, rank):
        self.dlraBlockInput.load_from_fullW(folder_name=folder_name, layer_id=0, rank=rank)
        self.dlraBlock1.load_from_fullW(folder_name=folder_name, layer_id=1, rank=rank)
        self.dlraBlock2.load_from_fullW(folder_name=folder_name, layer_id=2, rank=rank)
        self.dlraBlock3.load_from_fullW(folder_name=folder_name, layer_id=3, rank=rank)
        self.dlraBlockOutput.load(folder_name=folder_name, layer_id=4)
        return 0


class Linear2(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, name="linear", **kwargs):
        super(Linear2, self).__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal",
                                 trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear2, self).get_config()
        config.update({"units": self.units})
        return config

    def save(self, folder_name, layer_id):
        w_np = self.w.numpy()
        np.save(folder_name + "/w_" + str(layer_id) + ".npy", w_np)
        b_np = self.b.numpy()
        np.save(folder_name + "/b_" + str(layer_id) + ".npy", b_np)
        return 0

    def load(self, folder_name, layer_id):
        a_np = np.load(folder_name + "/w_" + str(layer_id) + ".npy")
        self.w = tf.Variable(initial_value=a_np,
                             trainable=True, name="w_", dtype=tf.float32)
        b_np = np.load(folder_name + "/b_" + str(layer_id) + ".npy")
        self.b = tf.Variable(initial_value=b_np,
                             trainable=True, name="b_", dtype=tf.float32)


class NaiveFactorLayer(keras.layers.Layer):
    def __init__(self, input_dim: int, units=32, low_rank=10, epsAdapt=0.1, rmax_total=100, name="dlra_block",
                 **kwargs):
        super(NaiveFactorLayer, self).__init__(**kwargs)
        self.epsAdapt = epsAdapt  # for unconventional integrator
        self.units = units
        self.low_rank = low_rank
        self.rmax_total = rmax_total
        self.input_dim = input_dim

    def build_model(self):
        self.k = self.add_weight(shape=(self.input_dim, self.low_rank), initializer="random_normal",
                                 trainable=True, name="k_")
        self.l_t = self.add_weight(shape=(self.low_rank, self.units), initializer="random_normal",
                                   trainable=True, name="lt_")
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True, name="b_")

        return 0

    @tf.function
    def call(self, inputs, step: int = 0):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """

        z = tf.matmul(tf.matmul(inputs, self.k), self.l_t) + self.b

        return tf.keras.activations.relu(z)

    def get_config(self):
        config = super(NaiveFactorLayer, self).get_config()
        config.update({"units": self.units})
        config.update({"low_rank": self.low_rank})
        return config

    def save(self, folder_name, layer_id):
        # main_variables
        k_np = self.k.numpy()
        np.save(folder_name + "/k" + str(layer_id) + ".npy", k_np)
        l_t_np = self.l_t.numpy()
        np.save(folder_name + "/l_t" + str(layer_id) + ".npy", l_t_np)
        b_np = self.b.numpy()
        np.save(folder_name + "/b" + str(layer_id) + ".npy", b_np)
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
        bias = np.load(folder_name + "/b" + str(layer_id) + ".npy")
        self.b = tf.Variable(initial_value=bias,
                             trainable=True, name="b_", dtype=tf.float32)

        return 0

    def load_from_fullW(self, folder_name, layer_id, rank):
        W_mat = np.load(folder_name + "/w_" + str(layer_id) + ".npy")
        d, u, v = tf.linalg.svd(W_mat)  # d=singular values, u2 = left singuar vecs, v2= right singular vecss

        s_init = tf.linalg.tensor_diag(d[:rank])
        u_init = u[:, :rank]
        v_init = v[:rank, :]
        self.k = tf.Variable(initial_value=u_init @ s_init, trainable=True, name="s_", dtype=tf.float32)
        self.l_t = tf.Variable(initial_value=v_init, trainable=True, name="Vt", dtype=tf.float32)

        return 0


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
