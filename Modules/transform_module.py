import tensorflow as tf
import tensorflow_transform as tft
import constants_for_transformer

_FLOAT_FEATURE = constants_for_transformer.FLOAT_FEATURE
_STRING_FEATURE = constants_for_transformer.STRING_FEATURE


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in _STRING_FEATURE:
        #print(key)
        #print(f"We want this line: {inputs[key]}")
        #_ = tft.vocabulary(inputs[key], vocab_filename=_STRING_FEATURE[0])
        outputs[key] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),top_k=50,num_oov_buckets=128)
        #outputs[key] = tf.cast(inputs[key],dtype=tf.uint64) #tft.vocabulary(inputs[key], vocab_filename=inputs[key])

        # and it must be passed in outputs to be passed to the model
       # print(f"look for this line: {outputs[key]}, {inputs[key]}")

    for key in _FLOAT_FEATURE:
        # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
        #outputs[key] = inputs[key] / 10
        outputs[key] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))

    return outputs


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)
