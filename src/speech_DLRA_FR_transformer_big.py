import networks.transformer_dlra_fr
import networks.transformer

import tensorflow as tf
import tensorflow_datasets as tfds

from optparse import OptionParser
from networks.utils import create_csv_logger_cb, list_of_lists_to_string, test_transformer

import time

# global constants # specify training
MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64
EPOCHS = 400

# global model
# Text tokenization & detokenization
model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)

# global network properties
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def train(low_rank):
    filename = "./logs/big_DLRA_FR_transformer_f/fix_rank_" + str(low_rank)
    filename_check = "./weight_checks/big_DLRA_FR_transformer_f/fix_rank_" + str(low_rank)

    # load dataset
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train_examples, val_examples, test_examples = examples['train'], examples['validation'], examples['test']

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
    print()
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

    # investigate data

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    num_layers = 6
    d_model = 512
    dff = 2048
    num_heads = 8
    dropout_rate = 0.1

    learning_rate = networks.transformer.CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    validation_accuracy = tf.keras.metrics.Mean(name='validation_accuracy')

    # build model
    transformer = networks.transformer_dlra_fr.TransformerDLRAFR(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        rate=dropout_rate,
        low_rank=low_rank)

    checkpoint_path = filename_check + '/checkpoints'

    # store model weights in checkpoints
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

        # Create logger
    log_file, file_name = create_csv_logger_cb(folder_name=filename)

    # print headline
    log_string = "epoch;time;loss_train;acc_train;loss_val;acc_val;compression;ranks\n"
    with open(file_name, "a") as log:
        log.write(log_string)

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step_low_rank(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        # 1.a) K and L Step Preproccessing
        transformer.k_step_preprocessing()
        transformer.l_step_preprocessing()

        # 1.b) Tape Gradients for K-Step
        # transformer.toggle_non_s_step_training()
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True, step=0)
            loss = loss_function(tar_real, predictions)

        # Gradient updates for k step
        grads_k_step = tape.gradient(loss, transformer.trainable_weights)
        transformer.set_none_grads_to_zero(grads_k_step, transformer.trainable_weights)
        # transformer.set_dlra_bias_grads_to_zero(grads_k_step)

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))

        # 1.b) Tape Gradients for L-Step
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True, step=1)
            loss = loss_function(tar_real, predictions)

        grads_l_step = tape.gradient(loss, transformer.trainable_weights)
        transformer.set_none_grads_to_zero(grads_l_step, transformer.trainable_weights)
        # transformer.set_dlra_bias_grads_to_zero(grads_l_step)

        # Gradient update for K and L
        optimizer.apply_gradients(zip(grads_k_step, transformer.trainable_weights))
        optimizer.apply_gradients(zip(grads_l_step, transformer.trainable_weights))

        # Postprocessing K and L
        transformer.k_step_postprocessing()
        transformer.l_step_postprocessing()

        # S-Step Preprocessing
        transformer.s_step_preprocessing()

        # transformer.toggle_s_step_training()

        # 3.b) Tape Gradients
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True, step=2)
            loss = loss_function(tar_real, predictions)

        # 3.c) Apply Gradients
        grads_s = tape.gradient(loss, transformer.trainable_weights)
        transformer.set_none_grads_to_zero(grads_s, transformer.trainable_weights)
        optimizer.apply_gradients(zip(grads_s, transformer.trainable_weights))  # All gradients except K and L matrix

        return 0

    @tf.function(input_signature=train_step_signature)
    def validation_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        predictions, _ = transformer([inp, tar_inp], training=False, step=2)

        loss = loss_function(tar_real, predictions)

        validation_loss(loss)
        validation_accuracy(accuracy_function(tar_real, predictions))

    ranks = []
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        validation_loss.reset_states()
        validation_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step_low_rank(inp, tar)
            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
                print("Ranks:")
                print(transformer.get_rank())

        # compute validation
        for (batch, (inp, tar)) in enumerate(val_batches):
            validation_step(inp, tar)

        # Log Data of current epoch
        log_string = str(epoch) + ";" + str(time.time() - start) + ";" + str(train_loss.result().numpy()) + ";" + str(
            train_accuracy.result().numpy()) + ";" + str(validation_loss.result().numpy()) + ";" + str(
            validation_accuracy.result().numpy()) + ";" + str(
            1 - transformer.get_compression_rate()) + list_of_lists_to_string(transformer.get_rank()) + "\n"

        with open(file_name, "a") as log:
            log.write(log_string)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    test_transformer(transformer, tokenizers, test_examples, filename, dlra=True)
    return 0


def filter_max_tokens(pt, en):
    num_tokens = tf.maximum(tf.shape(pt)[1], tf.shape(en)[1])
    return num_tokens < MAX_TOKENS


def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en


def make_batches(ds):
    return (
        ds
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(filter_max_tokens)
        .prefetch(tf.data.AUTOTUNE))


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


if __name__ == '__main__':
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-r", "--low_rank", dest="low_rank", default=50)
    parser.add_option("-e", "--epochs", dest="epochs", default=500)

    (options, args) = parser.parse_args()
    options.low_rank = int(options.low_rank)
    options.epochs = int(options.epochs)
    EPOCHS = options.epochs

    train(low_rank=options.low_rank)
