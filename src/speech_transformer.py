from networks.transformer import Translator

import tensorflow as tf
import numpy as np
from optparse import OptionParser
from os import path, makedirs
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

# global constants
MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64
EPOCHS = 20


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


def train(start_rank, tolerance, load_model, dim_layer, rmax, epochs):
    # specify training
    epochs = epochs
    batch_size = 256

    filename = "weight_data/transformer_sr" + str(start_rank) + "_v" + str(tolerance)
    folder_name = "weight_data/transformer" + str(start_rank) + "_v" + str(tolerance) + '/latest_model'
    folder_name_best = "weight_data/transformer" + str(start_rank) + "_v" + str(tolerance) + '/best_model'

    # check if dir exists
    if not path.exists(folder_name):
        makedirs(folder_name)
    if not path.exists(folder_name_best):
        makedirs(folder_name_best)

    print("save model as: " + filename)

    # load dataset
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))

    print()

    for en in en_examples.numpy():
        print(en.decode('utf-8'))

    # Text tokenization & detokenization
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )

    tokenizers = tf.saved_model.load(model_name)
    encoded = tokenizers.en.tokenize(en_examples)
    tokens = tokenizers.en.lookup(encoded)

    # investigate data
    lengths = []

    for pt_examples, en_examples in train_examples.batch(1024):
        pt_tokens = tokenizers.en.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())

        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)

    all_lengths = np.concatenate(lengths)

    plt.hist(all_lengths, np.linspace(0, 500, 101))
    plt.ylim(plt.ylim())
    max_length = max(all_lengths)
    plt.plot([max_length, max_length], plt.ylim())
    plt.title(f'Max tokens per example: {max_length}');
    plt.show()

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(d_model)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # build model
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        rate=dropout_rate)

    checkpoint_path = './checkpoints/train'

    # store model weights in checkpoints
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    return 0


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                     training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


if __name__ == '__main__':
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-s", "--start_rank", dest="start_rank", default=10)
    parser.add_option("-t", "--tolerance", dest="tolerance", default=10)
    parser.add_option("-l", "--load_model", dest="load_model", default=1)
    parser.add_option("-a", "--train", dest="train", default=1)
    parser.add_option("-d", "--dim_layer", dest="dim_layer", default=200)
    parser.add_option("-m", "--max_rank", dest="max_rank", default=200)
    parser.add_option("-e", "--epochs", dest="epochs", default=10)

    (options, args) = parser.parse_args()
    options.start_rank = int(options.start_rank)
    options.tolerance = float(options.tolerance)
    options.load_model = int(options.load_model)
    options.train = int(options.train)
    options.dim_layer = int(options.dim_layer)
    options.max_rank = int(options.max_rank)
    options.epochs = int(options.epochs)

    if options.train == 1:
        train(start_rank=options.start_rank, tolerance=options.tolerance, load_model=options.load_model,
              dim_layer=options.dim_layer, rmax=options.max_rank, epochs=options.epochs)
