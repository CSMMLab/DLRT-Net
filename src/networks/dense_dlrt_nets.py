import tensorflow as tf
from tensorflow import keras

from .dense_layers import Linear, DLRTLayer, DLRTLayerAdaptive


class DLRTNet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="e2eDLRANet", low_rank=20, dlra_layer_dim=200, **kwargs):
        super(DLRTNet, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.input_dim = input_dim
        self.dlra_layer_dim = dlra_layer_dim
        self.low_rank = low_rank
        self.output_dim = output_dim

        self.dlraBlockInput = DLRTLayer(input_dim=self.input_dim, units=self.dlra_layer_dim, low_rank=self.low_rank)
        self.dlraBlock1 = DLRTLayer(input_dim=self.dlra_layer_dim, units=self.dlra_layer_dim, low_rank=self.low_rank)
        self.dlraBlock2 = DLRTLayer(input_dim=self.dlra_layer_dim, units=self.dlra_layer_dim, low_rank=self.low_rank)
        self.dlraBlock3 = DLRTLayer(input_dim=self.dlra_layer_dim, units=self.dlra_layer_dim, low_rank=self.low_rank)
        self.dlraBlockOutput = Linear(input_dim=self.dlra_layer_dim, units=self.output_dim)

    def build_model(self):
        self.dlraBlockInput.build_model()
        self.dlraBlock1.build_model()
        self.dlraBlock2.build_model()
        self.dlraBlock3.build_model()
        self.dlraBlockOutput.build_model()

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


class DLRTNetAdaptive(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="e2eDLRANet", tol=0.4, low_rank=20, dlra_layer_dim=200,
                 rmax_total=100, **kwargs):
        super(DLRTNetAdaptive, self).__init__(name=name, **kwargs)
        # dlra_layer_dim = 250
        self.dlraBlockInput = DLRTLayerAdaptive(input_dim=input_dim, units=dlra_layer_dim, low_rank=low_rank,
                                                epsAdapt=tol,
                                                rmax_total=rmax_total, )
        self.dlraBlock1 = DLRTLayerAdaptive(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank,
                                            epsAdapt=tol,
                                            rmax_total=rmax_total, )
        self.dlraBlock2 = DLRTLayerAdaptive(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank,
                                            epsAdapt=tol,
                                            rmax_total=rmax_total, )
        self.dlraBlock3 = DLRTLayerAdaptive(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank,
                                            epsAdapt=tol,
                                            rmax_total=rmax_total, )
        self.dlraBlockOutput = Linear(input_dim=dlra_layer_dim, units=output_dim)

    def build_model(self):
        self.dlraBlockInput.build_model()
        self.dlraBlock1.build_model()
        self.dlraBlock2.build_model()
        self.dlraBlock3.build_model()
        self.dlraBlockOutput.build_model()

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


class ReferenceNet(keras.Model):

    def __init__(self, input_dim=10, output_dim=1, layer_dim=200, name="referenceNet", **kwargs):
        super(ReferenceNet, self).__init__(name=name, **kwargs)
        self.layer1 = Linear(units=layer_dim, input_dim=input_dim)
        self.layer2 = Linear(units=layer_dim, input_dim=layer_dim)
        self.layer3 = Linear(units=layer_dim, input_dim=layer_dim)
        self.layer4 = Linear(units=layer_dim, input_dim=layer_dim)
        self.layer5 = Linear(units=output_dim, input_dim=layer_dim)

    def build_model(self):
        self.layer1.build_model()
        self.layer2.build_model()
        self.layer3.build_model()
        self.layer4.build_model()
        self.layer5.build_model()
        return 0

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

    def save(self, folder_name):
        self.layer1.save(folder_name, layer_id=0)
        self.layer2.save(folder_name, layer_id=1)
        self.layer3.save(folder_name, layer_id=2)
        self.layer4.save(folder_name, layer_id=3)
        self.layer5.save(folder_name, layer_id=4)
        return

    def load(self, folder_name):
        self.layer1.load(folder_name=folder_name, layer_id=0)
        self.layer2.load(folder_name=folder_name, layer_id=1)
        self.layer3.load(folder_name=folder_name, layer_id=2)
        self.layer4.load(folder_name=folder_name, layer_id=3)
        self.layer5.load(folder_name=folder_name, layer_id=4)
        return 0
