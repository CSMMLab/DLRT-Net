import tensorflow as tf
from tensorflow import keras
import numpy as np


# Layers-----
class DLRALayerConv(keras.layers.Layer):
    def __init__(self, low_rank=10, epsAdapt=0.1, rmax_total=100, stride: tuple = (5, 5), rate: tuple = (2, 2),
                 size: tuple = (3, 3), filters=10, image_dims=(28, 28, 1), name="dlra_block_Conv2D",
                 **kwargs):
        super(DLRALayerConv, self).__init__(**kwargs)
        # DLRA options
        self.epsAdapt = epsAdapt  # for unconventional integrator
        self.low_rank = low_rank
        self.rmax_total = rmax_total

        # Convolution options
        self.stride = stride
        self.rate = rate
        self.filters = filters
        self.channels = image_dims[2]
        self.size = size
        self.image_dims = image_dims
        # Resulting shapes
        self.units = self.filters  # output dimension
        self.input_dim = self.size[0] * self.size[1] * self.channels

        self.rmax_total = min(self.rmax_total, int(min(self.units, self.input_dim) / 2))
        print("Max Rank has been set to:" + str(
            self.rmax_total) + " due to layer layout. Max allowed rank is min(in_dim,out_dim)/2")
        self.low_rank = min(self.low_rank, int(self.rmax_total))
        print("Start rank has been set to: " + str(self.low_rank) + " to match max rank")

        # Compute output patch shape
        batch_size = 4
        test_imgs = tf.ones(shape=(batch_size, image_dims[0], image_dims[1], image_dims[2]))

        patches = tf.image.extract_patches(images=test_imgs,
                                           sizes=[1, self.size[0], self.size[1], 1],
                                           strides=[1, self.stride[0], self.stride[1], 1],
                                           rates=[1, self.rate[0], self.rate[1], 1],
                                           padding='SAME')
        # patches dim: (batch,row,col,L), where L  = size[0]xsize[1]xC_in = self.input_dim
        # output dims are rowxcolxfilters
        print("Image patches for conv layer")
        print(patches.shape)
        # sanity check
        W = tf.ones(shape=(self.input_dim, self.filters))
        out = tf.tensordot(patches, W, axes=([-1], [0]))
        print(out.shape)
        self.output_shape_conv = (out.shape[1], out.shape[2], out.shape[3])
        print("Sanity check for conv layer passed")

    def build_model(self):

        self.k = self.add_weight(shape=(self.input_dim, self.low_rank), initializer="random_normal",
                                 trainable=True, name="k_")
        self.l_t = self.add_weight(shape=(self.low_rank, self.units), initializer="random_normal",
                                   trainable=True, name="lt_")
        self.s = self.add_weight(shape=(self.low_rank, self.low_rank), initializer="random_normal",
                                 trainable=True, name="s_")
        self.b = self.add_weight(shape=self.output_shape_conv, initializer="random_normal", trainable=True, name="b_")
        # auxiliary variables
        self.aux_b = self.add_weight(shape=self.output_shape_conv, initializer="random_normal", trainable=False,
                                     name="aux_b")
        self.aux_b.assign(self.b)  # non trainable bias or k and l step
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
        return 0

    # @tf.function
    def call(self, inputs, step: int = 0):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """
        # Convert Input in Patched Convolution
        patches = tf.image.extract_patches(images=inputs,
                                           sizes=[1, self.size[0], self.size[1], 1],
                                           strides=[1, self.stride[0], self.stride[1], 1],
                                           rates=[1, self.rate[0], self.rate[1], 1],
                                           padding='SAME')

        if step == 0:  # k-step
            z = tf.tensordot(patches, self.k, axes=([-1], [0]))
            z = tf.tensordot(z, self.aux_Vt, axes=([-1], [0]))
            z = z + self.aux_b
        elif step == 1:  # l-step
            z = tf.tensordot(patches, self.aux_U, axes=([-1], [0]))
            z = tf.tensordot(z, self.l_t, axes=([-1], [0]))
            z = z + self.aux_b
        else:  # s-step
            z = tf.tensordot(patches, self.aux_Unp1, axes=([-1], [0]))
            z = tf.tensordot(z, self.s, axes=([-1], [0]))
            z = tf.tensordot(z, self.aux_Vtnp1, axes=([-1], [0]))
            z = z + self.b
        return tf.keras.activations.relu(z)

    @tf.function
    def k_step_preprocessing(self, ):
        k = tf.matmul(self.aux_U, self.s)
        self.k.assign(k)  # = tf.Variable(initial_value=k, trainable=True, name="k_")
        return 0

    @tf.function
    def k_step_postprocessing(self):
        aux_Unp1, _ = tf.linalg.qr(self.k)
        self.aux_Unp1.assign(aux_Unp1)  # = tf.Variable(initial_value=aux_Unp1, trainable=False, name="aux_Unp1")
        N = tf.matmul(tf.transpose(self.aux_Unp1), self.aux_U)
        self.aux_N.assign(N)
        return 0

    @tf.function
    def l_step_preprocessing(self, ):
        l_t = tf.matmul(self.s, self.aux_Vt)
        self.l_t.assign(l_t)  # = tf.Variable(initial_value=l_t, trainable=True, name="lt_")
        return 0

    @tf.function
    def l_step_postprocessing(self):
        aux_Vtnp1, _ = tf.linalg.qr(tf.transpose(self.l_t))
        self.aux_Vtnp1.assign(tf.transpose(aux_Vtnp1))
        M = tf.matmul(self.aux_Vtnp1, tf.transpose(self.aux_Vt))
        self.aux_M.assign(M)
        return 0

    @tf.function
    def s_step_preprocessing(self):
        self.aux_U.assign(self.aux_Unp1)
        self.aux_Vt.assign(self.aux_Vtnp1)
        s = tf.matmul(tf.matmul(self.aux_N, self.s), tf.transpose(self.aux_M))
        self.s.assign(s)  # = tf.Variable(initial_value=s, trainable=True, name="s_")
        return 0

    def get_config(self):
        config = super(DLRALayerConv, self).get_config()
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


class DLRALayerConvAdaptive(keras.layers.Layer):
    def __init__(self, low_rank=10, epsAdapt=0.1, rmax_total=100, stride: tuple = (5, 5), rate: tuple = (2, 2),
                 size: tuple = (3, 3), filters=10, image_dims=(28, 28, 1), name="dlra_block_Conv2D",
                 **kwargs):
        super(DLRALayerConvAdaptive, self).__init__(**kwargs)
        # DLRA options
        self.epsAdapt = epsAdapt  # for unconventional integrator
        self.low_rank = low_rank  # min(image_dims[2] * size[0] * size[1], filters)
        self.rmax_total = rmax_total

        # Convolution options
        self.stride = stride
        self.rate = rate
        self.filters = filters
        self.channels = image_dims[2]
        self.size = size
        self.image_dims = image_dims
        # Resulting shapes
        self.units = self.filters  # output dimension
        self.input_dim = self.size[0] * self.size[1] * self.channels

        self.rmax_total = min(self.rmax_total, int(min(self.units, self.input_dim) / 2))
        print("Max Rank has been set to:" + str(
            self.rmax_total) + " due to layer layout. Max allowed rank is min(in_dim,out_dim)/2")
        self.low_rank = min(self.low_rank, int(self.rmax_total))
        print("Start rank has been set to: " + str(self.low_rank) + " to match max rank")

        # Compute output patch shape
        batch_size = 4
        test_imgs = tf.ones(shape=(batch_size, image_dims[0], image_dims[1], image_dims[2]))

        patches = tf.image.extract_patches(images=test_imgs,
                                           sizes=[1, self.size[0], self.size[1], 1],
                                           strides=[1, self.stride[0], self.stride[1], 1],
                                           rates=[1, self.rate[0], self.rate[1], 1],
                                           padding='SAME')
        # patches dim: (batch,row,col,L), where L  = size[0]xsize[1]xC_in = self.input_dim
        # output dims are rowxcolxfilters
        print("Image patches for conv layer")
        print(patches.shape)
        # sanity check
        W = tf.ones(shape=(self.input_dim, self.filters))
        out = tf.tensordot(patches, W, axes=([-1], [0]))
        print(out.shape)
        self.output_shape_conv = (out.shape[1], out.shape[2], out.shape[3])
        print("Sanity check for conv layer passed")

    def build_model(self):

        self.rmax_total = min(self.rmax_total, int(min(self.units, self.input_dim) / 2))
        print("Max Rank has been set to:" + str(
            self.rmax_total) + " due to layer layout. Max allowed rank is min(in_dim,out_dim)/2")
        self.low_rank = min(self.low_rank, int(self.rmax_total))
        print("Start rank has been set to: " + str(self.low_rank) + " to match max rank")
        self.input_dim = self.input_dim

        self.k = self.add_weight(shape=(self.input_dim, self.rmax_total), initializer="random_normal",
                                 trainable=True, name="k_")
        self.l_t = self.add_weight(shape=(self.rmax_total, self.units), initializer="random_normal",
                                   trainable=True, name="lt_")
        self.s = self.add_weight(shape=(2 * self.rmax_total, 2 * self.rmax_total), initializer="random_normal",
                                 trainable=True, name="s_")
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True, name="b_")
        # auxiliary variables
        self.aux_b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=False, name="aux_b")
        self.aux_b.assign(self.b)  # non trainable bias or k and l step

        self.aux_U = self.add_weight(shape=(self.input_dim, self.rmax_total), initializer="random_normal",
                                     trainable=False, name="aux_U")
        self.aux_Unp1 = self.add_weight(shape=(self.input_dim, 2 * self.rmax_total), initializer="random_normal",
                                        trainable=False, name="aux_Unp1")
        self.aux_Vt = self.add_weight(shape=(self.rmax_total, self.units), initializer="random_normal",
                                      trainable=False, name="Vt")
        self.aux_Vtnp1 = self.add_weight(shape=(2 * self.rmax_total, self.units), initializer="random_normal",
                                         trainable=False, name="vtnp1")
        self.aux_N = self.add_weight(shape=(2 * self.rmax_total, self.rmax_total), initializer="random_normal",
                                     trainable=False, name="aux_N")
        self.aux_M = self.add_weight(shape=(2 * self.rmax_total, self.rmax_total), initializer="random_normal",
                                     trainable=False, name="aux_M")
        # Todo: initializer with low rank
        return 0

    def call(self, inputs, step: int = 0):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """
        # Convert Input in Patched Convolution
        patches = tf.image.extract_patches(images=inputs,
                                           sizes=[1, self.size[0], self.size[1], 1],
                                           strides=[1, self.stride[0], self.stride[1], 1],
                                           rates=[1, self.rate[0], self.rate[1], 1],
                                           padding='SAME')

        if step == 0:  # k-step
            z = tf.tensordot(patches, self.k[:, :self.low_rank], axes=([-1], [0]))
            z = tf.tensordot(z, self.aux_Vt[:self.low_rank, :], axes=([-1], [0]))
            # z = tf.matmul(tf.matmul(inputs, self.k), self.aux_Vt)
            z = z + self.aux_b

        elif step == 1:  # l-step
            z = tf.tensordot(patches, self.aux_U[:, :self.low_rank], axes=([-1], [0]))
            z = tf.tensordot(z, self.l_t[:self.low_rank, :], axes=([-1], [0]))
            z = z + self.aux_b

            # z = tf.matmul(tf.matmul(inputs, self.aux_U), self.l_t)
        else:  # s-step
            z = tf.tensordot(patches, self.aux_Unp1[:, :2 * self.low_rank], axes=([-1], [0]))
            z = tf.tensordot(z, self.s[:2 * self.low_rank, :2 * self.low_rank], axes=([-1], [0]))
            z = tf.tensordot(z, self.aux_Vtnp1[:2 * self.low_rank, :], axes=([-1], [0]))
            # z = tf.matmul(tf.matmul(tf.matmul(inputs, self.aux_Unp1), self.s), self.aux_Vtnp1)
            z = z + self.b

        return tf.keras.activations.relu(z)

    # @tf.function
    def k_step_preprocessing(self):
        k = tf.matmul(self.aux_U[:, :self.low_rank], self.s[:self.low_rank, :self.low_rank])
        self.k[:, :self.low_rank].assign(k)
        return 0

    # @tf.function
    def k_step_postprocessing_adapt(self):
        k_extended = tf.concat((self.k[:, :self.low_rank], self.aux_U[:, :self.low_rank]), axis=1)
        aux_Unp1, _ = tf.linalg.qr(k_extended)
        self.aux_Unp1[:, :2 * self.low_rank].assign(aux_Unp1)
        aux_N = tf.matmul(tf.transpose(self.aux_Unp1[:, :2 * self.low_rank]), self.aux_U[:, : self.low_rank])
        self.aux_N[:2 * self.low_rank, :self.low_rank].assign(aux_N)
        return 0

    # @tf.function
    def l_step_preprocessing(self):
        l_t = tf.matmul(self.s[:self.low_rank, :self.low_rank], self.aux_Vt[:self.low_rank, :])
        self.l_t[:self.low_rank, :].assign(l_t)  # = tf.Variable(initial_value=l_t, trainable=True, name="lt_")
        return 0

    # @tf.function
    def l_step_postprocessing_adapt(self):
        l_extended = tf.concat(
            (tf.transpose(self.l_t[:self.low_rank, :]), tf.transpose(self.aux_Vt[:self.low_rank, :])), axis=1)
        aux_Vnp1, _ = tf.linalg.qr(l_extended)
        self.aux_Vtnp1[:2 * self.low_rank, :].assign(tf.transpose(aux_Vnp1))
        aux_M = tf.matmul(self.aux_Vtnp1[:2 * self.low_rank, :], tf.transpose(self.aux_Vt[: self.low_rank, :]))
        self.aux_M[:2 * self.low_rank, :self.low_rank].assign(aux_M)
        return 0

    # @tf.function
    def s_step_preprocessing(self):
        s = tf.matmul(
            tf.matmul(self.aux_N[:2 * self.low_rank, :self.low_rank], self.s[: self.low_rank, :self.low_rank]),
            tf.transpose(self.aux_M[:2 * self.low_rank, :self.low_rank]))
        self.s[:2 * self.low_rank, :2 * self.low_rank].assign(s)

        return 0

    # @tf.function
    def rank_adaption(self):
        # 1) compute SVD of S
        # d=singular values, u2 = left singuar vecs, v2= right singular vecs
        s_small = self.s[:2 * self.low_rank, :2 * self.low_rank]
        d, u2, v2 = tf.linalg.svd(s_small)

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
        self.s[:rmax, :rmax].assign(tf.linalg.tensor_diag(d[:rmax]))

        # update u and v
        self.aux_U[:, :rmax].assign(tf.matmul(self.aux_Unp1[:, :2 * self.low_rank], u2[:, :rmax]))
        self.aux_Vt[:rmax, :].assign(tf.matmul(v2[:rmax, :], self.aux_Vtnp1[:2 * self.low_rank, :]))
        self.low_rank = int(rmax)

        # update bias
        self.aux_b.assign(self.b)
        return 0

    def get_config(self):
        config = super(DLRALayerConvAdaptive, self).get_config()
        config.update({"units": self.units})
        config.update({"low_rank": self.low_rank})
        return config

    def save(self, folder_name, layer_id):
        # main_variables
        k_np = self.k[:, :self.low_rank].numpy()
        np.save(folder_name + "/k" + str(layer_id) + ".npy", k_np)
        l_t_np = self.l_t[:self.low_rank, :self.low_rank].numpy()
        np.save(folder_name + "/l_t" + str(layer_id) + ".npy", l_t_np)
        s_np = self.s[:2 * self.low_rank, :2 * self.low_rank].numpy()
        np.save(folder_name + "/s" + str(layer_id) + ".npy", s_np)
        b_np = self.b.numpy()
        np.save(folder_name + "/b" + str(layer_id) + ".npy", b_np)
        # aux_variables
        aux_U_np = self.aux_U[:, :self.low_rank].numpy()
        np.save(folder_name + "/aux_U" + str(layer_id) + ".npy", aux_U_np)
        aux_Unp1_np = self.aux_Unp1[:, :2 * self.low_rank].numpy()
        np.save(folder_name + "/aux_Unp1" + str(layer_id) + ".npy", aux_Unp1_np)
        aux_Vt_np = self.aux_Vt[:, :self.low_rank].numpy()
        np.save(folder_name + "/aux_Vt" + str(layer_id) + ".npy", aux_Vt_np)
        aux_Vtnp1_np = self.aux_Vtnp1[:, :2 * self.low_rank].numpy()
        np.save(folder_name + "/aux_Vtnp1" + str(layer_id) + ".npy", aux_Vtnp1_np)
        aux_N_np = self.aux_N[:2 * self.low_rank, :self.low_rank].numpy()
        np.save(folder_name + "/aux_N" + str(layer_id) + ".npy", aux_N_np)
        aux_M_np = self.aux_M[:2 * self.low_rank, :self.low_rank].numpy()
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


class LayerConv(keras.layers.Layer):
    def __init__(self, stride: tuple = (5, 5), rate: tuple = (2, 2),
                 size: tuple = (3, 3), filters=10, image_dims=(28, 28, 1), name="Conv2D",
                 **kwargs):
        super(LayerConv, self).__init__(**kwargs)

        # Convolution options
        self.stride = stride
        self.rate = rate
        self.filters = filters
        self.channels = image_dims[2]
        self.size = size
        self.image_dims = image_dims
        # Resulting shapes
        self.units = self.filters  # output dimension
        self.input_dim = self.size[0] * self.size[1] * self.channels

        # Compute output patch shape
        batch_size = 4
        test_imgs = tf.ones(shape=(batch_size, image_dims[0], image_dims[1], image_dims[2]))

        patches = tf.image.extract_patches(images=test_imgs,
                                           sizes=[1, self.size[0], self.size[1], 1],
                                           strides=[1, self.stride[0], self.stride[1], 1],
                                           rates=[1, self.rate[0], self.rate[1], 1],
                                           padding='SAME')
        # patches dim: (batch,row,col,L), where L  = size[0]xsize[1]xC_in = self.input_dim
        # output dims are rowxcolxfilters
        print("Image patches for conv layer")
        print(patches.shape)
        # sanity check
        W = tf.ones(shape=(self.input_dim, self.filters))
        out = tf.matmul(patches, W)

        # out = tf.tensordot(patches, W, axes=([-1], [0]))
        print(out.shape)
        self.output_shape_conv = (out.shape[1], out.shape[2], out.shape[3])
        print("Sanity check for conv layer passed")

    def build_model(self):
        self.W = self.add_weight(shape=(self.input_dim, self.units), initializer="random_normal",
                                 trainable=True, name="W_")
        self.b = self.add_weight(shape=self.output_shape_conv, initializer="random_normal", trainable=True, name="b_")

        return 0

    # @tf.function
    def call(self, inputs):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """
        # Convert Input in Patched Convolution
        patches = tf.image.extract_patches(images=inputs,
                                           sizes=[1, self.size[0], self.size[1], 1],
                                           strides=[1, self.stride[0], self.stride[1], 1],
                                           rates=[1, self.rate[0], self.rate[1], 1],
                                           padding='SAME')

        z = tf.matmul(patches, self.W)
        # z = tf.tensordot(patches, self.W, axes=([-1], [0]))
        return tf.keras.activations.relu(z + self.b)

    def save(self, folder_name, layer_id):
        # main_variables
        W = self.W.numpy()
        np.save(folder_name + "/W" + str(layer_id) + ".npy", W)

        b_np = self.b.numpy()
        np.save(folder_name + "/b" + str(layer_id) + ".npy", b_np)
        return 0

    def load(self, folder_name, layer_id):
        # main variables
        W = np.load(folder_name + "/k" + str(layer_id) + ".npy")
        self.W = tf.Variable(initial_value=W,
                             trainable=True, name="W_", dtype=tf.float32)
        b = np.load(folder_name + "/b" + str(layer_id) + ".npy")
        self.b = tf.Variable(initial_value=b,
                             trainable=True, name="b_", dtype=tf.float32)
        return 0
