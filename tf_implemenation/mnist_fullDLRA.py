from dlranet import ReferenceNet, FullDLRANet, create_csv_logger_cb

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from os import path, makedirs


def main3():
    
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-s", "--start_rank", dest="start_rank", default=10)
    parser.add_option("-t", "--tolerance", dest="tolerance", default=10)


    (options, args) = parser.parse_args()
    options.start_rank = int(options.start_rank)
    options.tolerance = float(options.tolerance)
  
    
    # specify training
    epochs = 2000
    batch_size = 256

    filename= "200x3_sr"+str(options.start_rank) + "_v"+ str(options.tolerance)
    folder_name= "200x3_sr"+str(options.start_rank) + "_v"+ str(options.tolerance) +   '/latest_model/'
    # check if dir exists
    if not path.exists(folder_name):
        makedirs(folder_name )
        
    print("save model as: " + filename)

    # Create Model
    input_dim = 784  # 28x28  pixel per image
    output_dim = 10  # one-hot vector of digits 0-9

    starting_rank = options.start_rank  #starting rank of S matrix
    tol = options.tolerance # eigenvalue treshold
    max_rank = 150 # maximum rank of S matrix

    dlra_layer_dim = 200
    model = FullDLRANet(input_dim=input_dim, output_dim=output_dim, low_rank=starting_rank,
                        dlra_layer_dim=dlra_layer_dim, tol=tol, rmax_total=max_rank)
    # Build optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()
    loss_metric_acc = tf.keras.metrics.Accuracy()

    # Build dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, input_dim))
    x_test = np.reshape(x_test, (-1, input_dim))

    # Reserve 10,000 samples for validation.
    #val_size = 1000
    #x_val = x_train[-val_size:]
    #y_val = y_train[-val_size:]
    #(x_val, y_val) = normalize_img(x_val, y_val)

    x_train = x_train#[:-val_size]
    y_train = y_train#[:-val_size]
    (x_train, y_train) = normalize_img(x_train, y_train)

    (x_val, y_val) = normalize_img(x_test, y_test)

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)

    # Create logger
    log_file, file_name = create_csv_logger_cb(folder_name=filename)

    # Iterate over epochs. (Training loop)
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
                # softmax activation for classification
                out = tf.keras.activations.softmax(out)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads_k_step = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_k_step, model.trainable_weights)
            model.set_dlra_bias_grads_to_zero(grads_k_step)

            # 1.b) Tape Gradients for L-Step
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=1, training=True)
                # softmax activation for classification
                out = tf.keras.activations.softmax(out)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads_l_step = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads_l_step, model.trainable_weights)
            model.set_dlra_bias_grads_to_zero(grads_l_step)

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
                # softmax activation for classification
                out = tf.keras.activations.softmax(out)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
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
            prediction = tf.math.argmax(out, 1)
            loss_metric_acc(prediction, batch_train[1])

            loss_value = loss_metric.result().numpy()
            acc_value = loss_metric_acc.result().numpy()
            if step % 100 == 0:
                print("step %d: mean loss S-Step = %.4f" % (step, loss_value))
                print("Accuracy: " + str(acc_value))
                print("Current Rank: " + str(int(model.dlraBlock1.low_rank)) + " | " + str(
                    int(model.dlraBlock2.low_rank)) + " | " + str(int(model.dlraBlock3.low_rank)))

        # Compute vallidation loss and accuracy
        loss_val = 0
        acc_val = 0

        #  K  Step Preproccessing 
        model.dlraBlock1.k_step_preprocessing()
        model.dlraBlock2.k_step_preprocessing()
        model.dlraBlock3.k_step_preprocessing()

        out = model(x_val, step=0, training=False)
        out = tf.keras.activations.softmax(out)
        loss = loss_fn(y_val, out)
        loss_metric(loss)
        loss_val = loss_metric.result()

        prediction = tf.math.argmax(out, 1)
        loss_metric_acc(prediction, y_val)
        acc_val = loss_metric_acc.result()

        # Log Data of current epoch
        log_string = str(loss_value) + ";" + str(acc_value) + ";" + str(
            loss_val.numpy()) + ";" + str(acc_val.numpy()) + ";" + str(loss_val.numpy()) + ";" + str(
            int(model.dlraBlock1.low_rank)) + ";" + str(int(model.dlraBlock2.low_rank)) + ";" + str(
            int(model.dlraBlock3.low_rank)) + "\n"
        with open(file_name, "a") as log:
            log.write(log_string)

        # save current model
        model.save(folder_name=folder_name)

    return 0

"""
def main2():
    # Create Model
    input_dim = 784  # 28x28  pixel per image
    output_dim = 10  # one-hot vector of digits 0-9
    model = ReferenceNet(output_dim=output_dim)  # input_dim gets automatically assigned
    # Build optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()

    # specify training
    epochs = 10
    batch_size = 64
    # Build dataset
    # Prepare the training dataset.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, input_dim))
    x_test = np.reshape(x_test, (-1, input_dim))

    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    (x_val, y_val) = normalize_img(x_val, y_val)

    t = x_val[0]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    (x_train, y_train) = normalize_img(x_train, y_train)

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                out = model(batch_train[0])
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    test = model(x_val)
    # plt.plot(test_x, test.numpy(), '-.')
    # plt.plot(test_x, test_y, '--')
    # plt.show()
    return 0
"""

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    # main()
    main3()
