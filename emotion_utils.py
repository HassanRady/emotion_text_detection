import tensorflow as tf
from tfx import v1 as tfx

from typing import List, Text
from absl import logging
import tensorflow_transform as tft
from tfx_bsl.public import tfxio

# Specify features that we will use.
_FEATURE_KEYS = [
    'text',
]
_FEATURE_KEY = 'text'
_LABEL_KEY = 'emotions'

_TRAIN_BATCH_SIZE = 128
_EVAL_BATCH_SIZE = 32
_VOCAB_SIZE = 6000
_MAX_LEN = 60


def _transformed_name(key, is_input=False):
    return key + ('_xf_input' if is_input else '_xf')


# NEW: TFX Transform will call this function.
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature.
    """
    outputs = {}
    review_sparse = tf.strings.split(tf.reshape(
        inputs[_FEATURE_KEY], [-1])).to_sparse()

    review_indices = tft.compute_and_apply_vocabulary(
        review_sparse, default_value=_VOCAB_SIZE, top_k=_VOCAB_SIZE)
    dense = tf.sparse.to_dense(review_indices, default_value=-1)
    # TFX transform expects the transform result to be FixedLenFeature.
    padding_config = [[0, 0], [0, _MAX_LEN]]
    dense = tf.pad(dense, padding_config, 'CONSTANT', -1)
    padded = tf.slice(dense, [0, 0], [-1, _MAX_LEN])
    padded += 1

    outputs[_transformed_name(_FEATURE_KEY, True)] = padded
    outputs[_LABEL_KEY] = inputs[_LABEL_KEY]
    return outputs


# NEW: This function will apply the same transform operation to training data
#      and serving requests.
def _apply_preprocessing(raw_features, tft_layer):
    transformed_features = tft_layer(raw_features)
    if _LABEL_KEY in raw_features:
        transformed_label = transformed_features.pop(_LABEL_KEY)
        return transformed_features, transformed_label
    else:
        return transformed_features, None


# NEW: This function will create a handler function which gets a serialized
#      tf.example, preprocess and run an inference with it.
def _get_serve_tf_examples_fn(model, tf_transform_output):
    # We must save the tft_layer to the model to ensure its assets are kept and
    # tracked.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(text):
        reshaped_text = tf.reshape(text, [-1, 1])
        transformed_features = model.tft_layer({"text": reshaped_text})
        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      data_accessor: DataAccessor for converting input to RecordBatch.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch

    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(batch_size=batch_size),
        schema=tf_transform_output.raw_metadata.schema)

    transform_layer = tf_transform_output.transform_features_layer()

    def apply_transform(raw_features):
        return _apply_preprocessing(raw_features, transform_layer)

    return dataset.map(apply_transform).repeat()


def _build_keras_model() -> tf.keras.Model:
    """Creates a DNN Keras model for classifying penguin data.

    Returns:
      A Keras Model.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=_VOCAB_SIZE+2, output_dim=100, name=_transformed_name(_FEATURE_KEY)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            100, dropout=0.3, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            100, dropout=0.3, return_sequences=True)),
        tf.keras.layers.Conv1D(100, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax'),
    ],
        name="Emotion_Model")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = _build_keras_model()

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    # NEW: Save a computation graph including transform layer.
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='text'))
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
