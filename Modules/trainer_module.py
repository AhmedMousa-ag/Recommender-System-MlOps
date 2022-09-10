from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tfx import v1 as tfx
from Utils.utils import load_config_file
import tensorflow_transform as tft

config_file = load_config_file()

_LABEL_KEY = config_file["train_args"]["label_key"]
_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10

epochs = config_file["train_args"]["epochs"]


def _gzip_reader_fn(filenames):
    '''Load compressed dataset

    Args:
      filenames - filenames of TFRecords to load

    Returns:
      TFRecordDataset loaded from the filenames
    '''

    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=epochs,
              batch_size=32) -> tf.data.Dataset:
    '''Create batches of features and labels from TF Records

    Args:
      file_pattern - List of files or patterns of file paths containing Example records.
      tf_transform_output - transform output graph
      num_epochs - Integer specifying the number of times to read through the dataset.
              If None, cycles through the dataset forever.
      batch_size - An int representing the number of records to combine in a single batch.

    Returns:
      A dataset of dict elements, (or a tuple of dict elements and label).
      Each dict maps feature keys to Tensor or SparseTensor objects.
    '''
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=_LABEL_KEY)

    return dataset


def _build_keras_model(hp) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying penguin data.
    Returns:
      A Keras Model.
    """

    hp_units = hp.get("units")
    activations_choices = hp.get("activation")
    hp_learning_rate = hp.get("learning_rate")

    input_desc = keras.layers.Input(shape=[1, ], dtype=tf.float32,
                                    name="Description")  # Input layer names must be the same as the feature names

    input_rating = keras.layers.Input(shape=[1, ], dtype=tf.float32,
                                      name='IMDb_Rating')  # Input layer names must be the same as the feature names

    embed_desc = keras.layers.Embedding(input_dim=6000, output_dim=124, mask_zero=True)(input_desc)
    bidir_layers = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(hp_units, activation=activations_choices, return_sequences=True))(embed_desc)

    x = tf.keras.layers.Dense(64, activation=activations_choices)(bidir_layers)
    x = tf.keras.layers.Flatten()(x)

    rating_layer = tf.keras.layers.Dense(64, activation=activations_choices)(input_rating)
    rating_layer = tf.keras.layers.Flatten()(rating_layer)
    x = tf.keras.layers.concatenate([x, rating_layer])
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=[input_desc, input_rating], outputs=outputs)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    model.summary(print_fn=logging.info)
    return model


def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """

    # This schema is usually either an output of SchemaGen or a manually-curated
    # version provided by pipeline author. A schema can also derived from TFT
    # graph if a Transform component is used. In the case when either is missing,
    # `schema_from_feature_spec` could be used to generate schema from very simple
    # feature_spec, but the schema returned would be very primitive.

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')
    patience = config_file["train_args"]["early_stp_patience"]
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                           restore_best_weght=True)

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Use _input_fn() to extract input features and labels from the train and val set
    epochs = config_file["train_args"]["epochs"]  # Define our epochs here
    train_dataset = _input_fn(fn_args.train_files[0], tf_transform_output, epochs)
    eval_dataset = _input_fn(fn_args.eval_files[0], tf_transform_output, epochs)

    hp = fn_args.hyperparameters.get('values')
    model = _build_keras_model(hp=hp)
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stop_callback])

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf')
