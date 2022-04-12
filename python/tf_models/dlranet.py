'''
brief: Reference_model
Author: Steffen Schotthöfer
Version: 0.0
Date 12.04.2022
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PartDLRANet(keras.Model):

    def __init__(self, name="partDLRANet", **kwargs):
        super(PartDLRANet, self).__init__(name=name, **kwargs)
        self.denseBlock = DenseBlock()
        self.dlraBlock = DLRALayer(units=250, low_rank=100)
        self.outputBlock = DenseBlockOutput()

    def call(self, inputs, step: int):
        z = self.denseBlock(inputs)
        z = self.dlraBlock(z, step=step)
        z = self.outputBlock(z)
        return z


class DenseBlock(keras.layers.Layer):
    def __init__(self, name="dense_block", **kwargs):
        super(DenseBlock, self).__init__(name=name, **kwargs)
        self.layer1 = Linear(units=64)

    def call(self, inputs):
        z = self.layer1(inputs)
        z = tf.keras.activations.relu(z)
        return z


class DenseBlockOutput(keras.layers.Layer):
    def __init__(self, name="dense_output_block", **kwargs):
        super(DenseBlockOutput, self).__init__(name=name, **kwargs)
        self.layer3 = Linear(units=250)
        self.layer4 = Linear(units=1)

    def call(self, inputs):
        z = tf.keras.activations.relu(inputs)
        z = self.layer3(z)
        z = tf.keras.activations.relu(z)
        z = self.layer4(z)
        return z


class DLRALayer(keras.layers.Layer):
    def __init__(self, units=32, low_rank=10, name="dlra_block", **kwargs):
        super(DLRALayer, self).__init__(**kwargs)
        self.units = units
        self.low_rank = low_rank

    def build(self, input_shape):
        self.k = self.add_weight(shape=(input_shape[-1], self.low_rank), initializer="random_normal",
                                 trainable=True, )
        self.l_t = self.add_weight(shape=(self.units, self.low_rank), initializer="random_normal",
                                   trainable=True, )
        self.s = self.add_weight(shape=(self.low_rank, self.low_rank), initializer="random_normal",
                                 trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        # auxiliary variables
        self.aux_U = self.add_weight(shape=(self.units, self.low_rank), initializer="random_normal",
                                     trainable=False, )
        self.aux_Unp1 = self.add_weight(shape=(self.units, self.low_rank), initializer="random_normal",
                                        trainable=False, )
        self.aux_Vt = self.add_weight(shape=(self.low_rank, self.units), initializer="random_normal",
                                      trainable=False, )
        self.aux_Vtnp1 = self.add_weight(shape=(self.low_rank, self.units), initializer="random_normal",
                                         trainable=False, )
        self.aux_N = self.add_weight(shape=(self.low_rank, self.low_rank), initializer="random_normal",
                                     trainable=False, )
        self.aux_M = self.add_weight(shape=(self.low_rank, self.low_rank), initializer="random_normal",
                                     trainable=False, )

    def call(self, inputs, step):
        """
        :param inputs: layer input
        :param step: step conter: k:= 0, l:=1, s:=2
        :return:
        """
        if step == 0:  # k-step
            z = tf.matmul(inputs, tf.matmul(self.k, self.aux_Vt))
        elif step == 1:  # l-step
            z = tf.matmul(inputs, tf.matmul(self.aux_U, self.l_t))
        else:  # s-step
            z = tf.matmul(inputs, tf.matmul(self.aux_Unp1, tf.matmul(self.s, self.aux_Vtnp1)))
        return z

    def k_step_preprocessing(self, ):
        k = tf.matmul(self.aux_U, self.s)
        self.k = tf.Variable(initial_value=k, trainable=True, )
        return 0

    def k_step_postprocessing(self):
        aux_Unp1, _ = tf.linalg.qr(self.k)
        self.aux_Unp1 = tf.Variable(initial_value=aux_Unp1, trainable=False, )
        self.aux_N = tf.matmul(tf.transpose(self.aux_Unp1), self.aux_U)
        return 0

    def l_step_preprocessing(self, ):
        l_t = tf.matmul(self.s, self.aux_Vt)
        self.l_t = tf.Variable(initial_value=l_t, trainable=True, )
        return 0

    def l_step_postpocessing(self):
        aux_Vtnp1, _ = tf.linalg.qr(tf.transpose(self.l_t))
        self.aux_Vtnp1 = tf.transpose(aux_Vtnp1)
        self.aux_M = tf.matmul(self.aux_Vtnp1, tf.transpose(self.aux_Vt))
        return 0

    def s_step_preprocessing(self):
        s = tf.matmul(tf.matmul(self.aux_N, self.s), tf.transpose(self.aux_M))
        self.s = tf.Variable(initial_value=s, trainable=True, )
        return 0

    def s_step_postprocessing(self):
        self.aux_U = self.aux_Unp1
        self.aux_Vt = self.aux_Vtnp1
        return 0

    def get_config(self):
        config = super(DLRALayer, self).get_config()
        config.update({"units": self.units})
        config.update({"low_rank": self.low_rank})
        return config


class ReferenceNet(keras.Model):

    def __init__(self, name="referenceNet", **kwargs):
        super(ReferenceNet, self).__init__(name=name, **kwargs)
        self.layer1 = Linear(units=64)
        self.layer2 = Linear(units=250)
        self.layer3 = Linear(units=250)
        self.layer4 = Linear(units=1)

    def call(self, inputs):
        z = self.layer1(inputs)
        z = tf.keras.activations.relu(z)
        z = self.layer2(z)
        z = tf.keras.activations.relu(z)
        z = self.layer3(z)
        z = tf.keras.activations.relu(z)
        z = self.layer4(z)
        return z


class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal",
                                 trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config


# ------------------- Basisc example below ---------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
            self,
            original_dim,
            intermediate_dim=64,
            latent_dim=32,
            name="autoencoder",
            **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
