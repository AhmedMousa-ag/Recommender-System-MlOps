from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_KEYS = [
    'Title', 'Description', 'IMDb Rating'
]

_LABEL_KEY = 'My Rate'
_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      schema: schema of the input data.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_LABEL_KEY),
        schema=schema).repeat()


def _build_uni_embedding(uri="https://tfhub.dev/google/universal-sentence-encoder/4"):
    return hub.KerasLayer(uri, trainable=False)


"""
The model will consist of:
    1- Embedding layer for each string future, *- And an input layer for
    2- A Bidirectional layer for each Embedding
    3- A concatenate for the Bidirectional layers and the IMDB rating Dense Layer
    4- an Output layer of 1 neuron with a linear activation function 
"""


def _build_keras_model(hp) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying penguin data.
    Returns:
      A Keras Model.
    """
    hp_units = hp.get('units')
    activations_choices = hp.get('activation')

    embedding_model = _build_uni_embedding()
    embedding_model = _build_uni_embedding()
    inputs = [keras.layers.Input(shape=[], dtype=tf.string, name=f) for f in _FEATURE_KEYS]
    rating_inp = keras.layers.Input(shape=[1, ], dtype=tf.string, name='input_rating')
    embeddings = [embedding_model(inp) for inp in inputs[:-1]]
    embeddings = [tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(embed) for embed in embeddings]
    bidir_layers = [
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(hp_units, activation=activations_choices, return_sequences=True))(embed)
        for embed in embeddings]
    x = tf.keras.layers.concatenate(bidir_layers)
    x = tf.keras.layers.Dense(hp_units, activation=activations_choices)(x)
    rating_layer = tf.keras.layers.Dense(hp_units, activation=activations_choices)(rating_inp)
    rating_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 0))(rating_layer)
    x = tf.keras.layers.concatenate([x, rating_layer])
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=(inputs, rating_inp), outputs=outputs)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.get('learning_rate')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[keras.metrics.MeanSquaredError()])

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

    # TODO see how to replace it with our generated schema
    # schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema=fn_args.schema_file,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema=fn_args.schema_file,
        batch_size=_EVAL_BATCH_SIZE)

    model = _build_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf')
