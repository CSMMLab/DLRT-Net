from xmlrpc.client import boolean
from dlranetConvolution import DLRANetConv, create_csv_logger_cb

import tensorflow as tf
from tensorflow import keras

import numpy as np
from optparse import OptionParser
from os import path, makedirs


def test(start_rank, tolerance):
    # specify training
    folder_name = "e2edense_sr" + str(start_rank) + "_v" + str(tolerance) + '/latest_model'
    # check if dir exists
    if not path.exists(folder_name):
        print("error, file not found")
        exit(1)
    print("Load model from: " + folder_name)

    # Create Model
    input_dim = 784  # 28x28  pixel per image
    output_dim = 10  # one-hot vector of digits 0-9

    starting_rank = options.start_rank  # starting rank of S matrix
    tol = options.tolerance  # eigenvalue treshold
    max_rank = 150  # maximum rank of S matrix

    dlra_layer_dim = 200
    model = DLRANetConv(input_dim=input_dim, output_dim=output_dim, low_rank=starting_rank,
                        dlra_layer_dim=dlra_layer_dim, tol=tol, rmax_total=max_rank)
    # Build optimizer
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Accuracy()

    # Load model
    # if options.load_model:
    model.load(folder_name=folder_name)

    # Load dataset
    # Build dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x_train = np.reshape(x_train, (-1, input_dim))
    x_test = np.reshape(x_test, (-1, input_dim))

    (x_test, y_test) = normalize_img(x_test, y_test)

    # Test model
    #  K  Step Preproccessing
    # model.dlraBlock1.k_step_preprocessing()
    # model.dlraBlock2.k_step_preprocessing()
    # model.dlraBlock3.k_step_preprocessing()

    out = model(x_test, step=0, training=False)
    out = tf.keras.activations.softmax(out)
    loss = loss_fn(y_test, out)
    loss_metric(loss)
    loss_test = loss_metric.result().numpy()

    prediction = tf.math.argmax(out, 1)
    acc_metric.update_state(prediction, y_test)
    for pred, test in zip(prediction.numpy(), y_test):
        print("(" + str(pred) + "|" + str(test) + ")")

    acc_test = acc_metric.result().numpy()
    print("test Accuracy: " + str(acc_test))
    print("test loss: " + str(loss_test))
    print("Ranks")
    print(model.dlraBlock1.s.shape)
    print(model.dlraBlock2.s.shape)
    print(model.dlraBlock3.s.shape)
    acc_metric.reset_state()

    acc_metric.update_state([[1], [2]], [[0], [2]])
    print(acc_metric.result())
    return 0


def train(start_rank, tolerance, load_model):
    # specify training
    epochs = 100
    batch_size = 256

    filename = "e2edense_sr" + str(start_rank) + "_v" + str(tolerance)
    folder_name = "e2edense_sr" + str(start_rank) + "_v" + str(tolerance) + '/latest_model'
    folder_name_best = "e2edense_sr" + str(start_rank) + "_v" + str(tolerance) + '/best_model'

    # check if dir exists
    if not path.exists(folder_name):
        makedirs(folder_name)
    if not path.exists(folder_name_best):
        makedirs(folder_name_best)

    print("save model as: " + filename)

    # Create Model
    input_dim = 784  # 28x28  pixel per image
    output_dim = 10  # one-hot vector of digits 0-9

    starting_rank = start_rank  # starting rank of S matrix
    tol = tolerance  # eigenvalue treshold
    max_rank = 20  # maximum rank of S matrix

    dlra_layer_dim = 784
    model = DLRANetConv(low_rank=4, tol=tol, rmax_total=max_rank, image_dims=(32, 32, 3), output_dim=10)
    # Build optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Accuracy()
    loss_metric_acc_val = tf.keras.metrics.Accuracy()

    # Build dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]])
    # x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]])
    # x_train = np.reshape(x_train, (-1, 28, 28, 1))
    # x_test = np.reshape(x_test, (-1, 28, 28, 1))

    # Reserve 10,000 samples for validation.
    val_size = 10000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    (x_val, y_val) = normalize_img(x_val, y_val)

    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    (x_train, y_train) = normalize_img(x_train, y_train)

    (x_test, y_test) = normalize_img(x_test, y_test)
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Create logger
    log_file, file_name = create_csv_logger_cb(folder_name=filename)

    # print headline
    log_string = "loss_train;acc_train;loss_val;acc_val;loss_test;acc_test;rank1;rank2;rank3;rank4\n"
    with open(file_name, "a") as log:
        log.write(log_string)

    # load weights
    if load_model == 1:
        model.load(folder_name=folder_name)
    model.build_model()

    best_acc = 0
    best_loss = 10
    # Iterate over epochs. (Training loop)
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.

        for step, batch_train in enumerate(train_dataset):
            # 1.a) K and L Step Preproccessing
            model.k_step_preprocessing()
            model.l_step_preprocessing()

            # 1.b) Tape Gradients for K-Step
            model.toggle_non_s_step_training()
            with tf.GradientTape() as tape:
                out = model(batch_train[0], step=0, training=True)
                # softmax activation for classification
                out = tf.keras.activations.softmax(out)
                out = tf.reshape(out, shape=(batch_train[0].shape[0], 10))
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
            model.k_step_postprocessing()
            model.l_step_postprocessing()

            # S-Step Preprocessing
            model.s_step_preprocessing()
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

            # Network monotoring and verbosity
            loss_metric.update_state(loss)
            prediction = tf.math.argmax(out, 1)
            acc_metric.update_state(prediction, batch_train[1])

            loss_value = loss_metric.result().numpy()
            acc_value = acc_metric.result().numpy()
            if step % 2 == 0:
                print("step %d: mean loss S-Step = %.4f" % (step, loss_value))
                print("Accuracy: " + str(acc_value))
                print("Ranks:")
                print(model.get_low_ranks())

            # Reset metrics
            loss_metric.reset_state()
            acc_metric.reset_state()

        # Compute vallidation loss and accuracy
        loss_val = 0
        acc_val = 0

        #  K  Step Preproccessing
        model.k_step_preprocessing()

        # Validate model
        out = model(x_val, step=0, training=False)
        out = tf.keras.activations.softmax(out)
        loss = loss_fn(y_val, out)
        loss_metric.update_state(loss)
        loss_val = loss_metric.result().numpy()

        prediction = tf.math.argmax(out, 1)
        acc_metric.update_state(prediction, y_val)
        acc_val = acc_metric.result().numpy()
        print("Val Accuracy: " + str(acc_val))

        # save current model if it's the best
        if acc_val >= best_acc and loss_val <= best_loss:
            best_acc = acc_val
            best_loss = loss_val
            print("new best model with accuracy: " + str(best_acc) + " and loss " + str(best_loss))

            model.save(folder_name=folder_name_best)
        model.save(folder_name=folder_name)

        # Reset metrics
        loss_metric.reset_state()
        acc_metric.reset_state()

        # Test model
        out = model(x_test, step=0, training=False)
        out = tf.keras.activations.softmax(out)
        loss = loss_fn(y_test, out)
        loss_metric.update_state(loss)
        loss_test = loss_metric.result().numpy()

        prediction = tf.math.argmax(out, 1)
        acc_metric.update_state(prediction, y_test)
        acc_test = acc_metric.result().numpy()
        log_string = "Loss: " + str(loss_test) + "| Accuracy" + str(acc_test) + "\n"
        print("Test :" + log_string)
        # Reset metrics
        loss_metric.reset_state()
        acc_metric.reset_state()

        # Log Data of current epoch
        log_string = str(loss_value) + ";" + str(acc_value) + ";" + str(
            loss_val) + ";" + str(acc_val) + ";" + str(
            loss_test) + ";" + str(acc_test)
        ranks = model.get_low_ranks()
        for rank in ranks:
            log_string += ";" + str(rank)
        # + str(
        # int(model.dlraBlockInput.low_rank)) + ";" + str(
        # int(model.dlraBlock1.low_rank)) + ";" + str(int(model.dlraBlock2.low_rank)) + ";" + str(
        # int(model.dlraBlock3.low_rank)) + "\n"
        with open(file_name, "a") as log:
            log.write(log_string)
        print("Epoch Data :" + log_string)

    return 0


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-s", "--start_rank", dest="start_rank", default=10)
    parser.add_option("-t", "--tolerance", dest="tolerance", default=10)
    parser.add_option("-l", "--load_model", dest="load_model", default=1)
    parser.add_option("-a", "--train", dest="train", default=0)

    (options, args) = parser.parse_args()
    options.start_rank = int(options.start_rank)
    options.tolerance = float(options.tolerance)
    options.load_model = int(options.load_model)
    options.train = int(options.train)

    if options.train == 1:
        train(start_rank=options.start_rank, tolerance=options.tolerance, load_model=options.load_model)
    else:
        test(start_rank=options.start_rank, tolerance=options.tolerance)
