from dlranet import VariationalAutoEncoder, ReferenceNet, PartDLRANet, FullDLRANet

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def main3():
    # Create Model
    model = FullDLRANet(tol=0.05, low_rank=40, rmax_total=100)
    # model = PartDLRANet(tol=0.05, low_rank=40, rmax_total=100)

    # Build optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Choose loss
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()

    # Build dataset
    n_train = 10000
    n_test = 10000

    train_x = np.linspace(-2, 2, n_train).reshape((n_train, 1))
    train_y = np.square(train_x)

    test_x = np.linspace(-2, 2, n_test).reshape((n_test, 1))
    test_y = np.square(test_x)

    # Choose batch size
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # setup training
    epochs = 10

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.

        for step, batch_train in enumerate(train_dataset):
            # 1.a) K and L Step Preproccessing
            model.dlraBlock1.k_step_preprocessing()
            model.dlraBlock1.l_step_preprocessing()
            model.dlraBlock2.k_step_preprocessing()
            model.dlraBlock2.l_step_preprocessing()
            model.dlraBlock3.k_step_preprocessing()
            model.dlraBlock3.l_step_preprocessing()
            # 1.b) Tape Gradients for K-Step
            model.toggle_non_s_step_training()
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=0, training=True)
                # Compute reconstruction loss
                loss = mse_loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads_k_step = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_k_step, model.trainable_weights)
            grads_k_step[1] = grads_k_step[1] * 0  # Set bias grads to 0
            grads_k_step[5] = grads_k_step[5] * 0  # Set bias grads to 0
            grads_k_step[9] = grads_k_step[9] * 0  # Set bias grads to 0

            # 1.b) Tape Gradients for L-Step
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=1, training=True)
                # Compute reconstruction loss
                loss = mse_loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads_l_step = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_l_step, model.trainable_weights)
            grads_l_step[1] = grads_l_step[1] * 0  # Set bias grads to 0
            grads_l_step[5] = grads_l_step[5] * 0  # Set bias grads to 0
            grads_l_step[9] = grads_l_step[9] * 0  # Set bias grads to 0

            # Gradient update for K and L
            optimizer.apply_gradients(zip(grads_k_step, model.trainable_weights))
            optimizer.apply_gradients(zip(grads_l_step, model.trainable_weights))

            # Postprocessing K and L
            model.dlraBlock1.k_step_postprocessing_adapt()
            model.dlraBlock1.l_step_postprocessing_adapt()
            model.dlraBlock2.k_step_postprocessing_adapt()
            model.dlraBlock2.l_step_postprocessing_adapt()
            model.dlraBlock3.k_step_postprocessing_adapt()
            model.dlraBlock3.l_step_postprocessing_adapt()

            # S-Step Preprocessing
            model.dlraBlock1.s_step_preprocessing()
            model.dlraBlock2.s_step_preprocessing()
            model.dlraBlock3.s_step_preprocessing()

            model.toggle_s_step_training()

            # 3.b) Tape Gradients
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=2, training=True)
                # Compute reconstruction loss
                loss = mse_loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            # 3.c) Apply Gradients
            grads_s = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_s, model.trainable_weights)
            optimizer.apply_gradients(zip(grads_s, model.trainable_weights))  # All gradients except K and L matrix

            # Rank Adaptivity
            model.dlraBlock1.rank_adaption()
            model.dlraBlock2.rank_adaption()
            model.dlraBlock3.rank_adaption()

            # Network monotoring and verbosity
            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss S-Step = %.4f" % (step, loss_metric.result()))
                print("Current Rank: " + str(int(model.dlraBlock1.low_rank)) + " | " + str(
                    int(model.dlraBlock2.low_rank)) + " | " + str(int(model.dlraBlock3.low_rank)))

    test = model(test_x, step=0)
    plt.plot(test_x, test.numpy(), '-.')
    plt.plot(test_x, test_y, '--')
    plt.savefig("res_quad.png")
    return 0


def main2():
    original_dim = 784
    model = ReferenceNet()  # Build Model

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()

    # Build dataset 
    n_train = 10000
    n_test = 10000

    train_x = np.linspace(-2, 2, n_train).reshape((n_train, 1))
    train_y = np.square(train_x)

    test_x = np.linspace(-2, 2, n_test).reshape((n_test, 1))
    test_y = np.square(test_x)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)

    epochs = 10

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                out = model(batch_train[0])
                # Compute reconstruction loss
                loss = mse_loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    test = model(test_x)
    plt.plot(test_x, test.numpy(), '-.')
    plt.plot(test_x, test_y, '--')
    plt.show()
    return 0


def main():
    original_dim = 784
    vae = VariationalAutoEncoder(original_dim, 64, 32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    epochs = 2

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    return 0


if __name__ == '__main__':
    # main()
    main3()
