from typing import NamedTuple, Dict, Text, Any

import keras_tuner as kt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from tensorflow import keras
from tfx.components.trainer.fn_args_utils import FnArgs
from Utils.utils import load_config_file

config_file = load_config_file()

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

# Label key
LABEL_KEY = 'My Rate'

# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# -----------------------------------------------------------------------------------------
# The Transform component earlier saved the transformed examples as TFRecords compressed in .gz format.
# and you will need to load that into memory.
def _gzip_reader_fn(filenames):
    """Load compressed dataset

    Args:
      filenames - filenames of TFRecords to load

    Returns:
      TFRecordDataset loaded from the filenames
    """

    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


# -----------------------------------------------------------------------------------------
def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=6) -> tf.data.Dataset:
    """Create batches of features and labels from TF Records

    Args:
      file_pattern - List of files or patterns of file paths containing Example records.
      tf_transform_output - transform output graph
      num_epochs - Integer specifying the number of times to read through the dataset.
              If None, cycles through the dataset forever.
      batch_size - An int representing the number of records to combine in a single batch.

    Returns:
      A dataset of dict elements, (or a tuple of dict elements and label).
      Each dict maps feature keys to Tensor or SparseTensor objects.
    """
    # Get feature specification based on transform output
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # Create batches of features and labels
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=LABEL_KEY)

    return dataset


# -----------------------------------------------------------------------------------------


def model_builder(hp):
    """
    Builds the model and sets up the hyperparameters to tune.

    Args:
      hp - Keras tuner object

    Returns:
      model with hyperparameters to tune
    """

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    activations_choices = hp.Choice('activation', values=['relu', 'selu'])

    input_desc = keras.layers.Input(shape=[1, ], dtype=tf.string,
                                    name="Description")  # Input layer names must be the same as the feature names

    input_rating = keras.layers.Input(shape=[1, ], dtype=tf.float32,
                                      name='IMDb_Rating')  # Input layer names must be the same as the feature names
    squeezed = keras.layers.Lambda(lambda x: tf.squeeze(x))(input_desc)
    embed_desc = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)
    embed_desc = embed_desc(squeezed)

    expanded_layer = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(embed_desc)
    bidir_layers = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(hp_units, activation=activations_choices, return_sequences=True))(expanded_layer)

    x = tf.keras.layers.Dense(64, activation=activations_choices)(bidir_layers)
    x = tf.keras.layers.Flatten()(x)

    rating_layer = tf.keras.layers.Dense(64, activation=activations_choices)(input_rating)
    rating_layer = tf.keras.layers.Flatten()(rating_layer)
    x = tf.keras.layers.concatenate([x, rating_layer])
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=[input_desc, input_rating], outputs=outputs)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    return model


# -----------------------------------------------------------------------------------------
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.
    Args:
      fn_args: Holds args as name/value pairs.
        - working_dir: working dir for tuning.
        - train_files: List of file paths containing training tf.Example data.
        - eval_files: List of file paths containing eval tf.Example data.
        - train_steps: number of train steps.
        - eval_steps: number of eval steps.
        - schema_path: optional schema of the input data.
        - transform_graph_path: optional transform graph produced by TFT.
    Returns:
      A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                      model , e.g., the training and validation dataset. Required
                      args depend on the above tuner's implementation.
    """
    # Define tuner search strategy
    tuner = kt.Hyperband(model_builder,
                         objective='val_mean_absolute_error',
                         max_epochs=10,
                         factor=3,
                         directory=fn_args.working_dir,
                         project_name='kt_hyperband')

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Use _input_fn() to extract input features and labels from the train and val set
    train_set = _input_fn(fn_args.train_files[0], tf_transform_output)
    val_set = _input_fn(fn_args.eval_files[0], tf_transform_output)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [stop_early],
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
