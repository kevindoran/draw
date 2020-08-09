import tensorflow as tf
import tensorflow_datasets as tfds

import logging


DATA_DIR = 'data/mnist'
MODEL_DIR = 'out/models/mnist_model'
BATCH_SIZE = 128
TRAINING_STEPS = 10000
learning_rate_base = 0.05
TRAIN_SPLIT = 'train[0%:80%]'
EVAL_SPLIT = 'train[80%:100%]'


def mnist_ds(split, batch_size):
    def normalize_and_shape(img, label):
        # TODO: decide if the model or ds is to do this.
        img = tf.cast(img, tf.float32) / 255.0
        # img = tf.cast(img, tf.float32) 
        # Make 2D from 1D.
        img = tf.reshape(img, [28, 28])
        return (img, label)

    ds = tfds.load('mnist', split=split, shuffle_files=True,
            data_dir='./data/mnist', as_supervised=True)
    # ds = ds.map( lambda img, label: (tf.cast(img, tf.float32) / 255., label),       
    ds = ds.map(normalize_and_shape, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Repeat indefinitely, shuffle, and make batches.
    ds = ds.repeat().shuffle(buffer_size=10000).batch(BATCH_SIZE)
    return ds


def create_model(data_format='channels_first'):
    """Model to recognize MNIST digits.
  
    Args:
      data_format: Either 'channels_first' or 'channels_last'. 'channels_first' 
          is typically faster on GPUs while 'channels_last' is typically faster 
          on CPUs. See
          https://www.tensorflow.org/performance/performance_guide#data_formats
  
    Returns:
      A tf.keras.Model.
    """
    if data_format == 'channels_first':
      input_shape = [1, 28, 28]
    else:
      assert data_format == 'channels_last'
      input_shape = [28, 28, 1]
  
    l = tf.keras.layers
    max_pool = l.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    model = tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape, input_shape=(28, 28,)),
            l.Conv2D(
                32, 5, padding='same', data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Conv2D(
                64, 5, padding='same', data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(10, activation='softmax')
        ])
    return model


def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}


def compile_model():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

def train():
    estimator = tf.keras.estimator.model_to_estimator(
            keras_model=compile_model(), model_dir=MODEL_DIR)
    tf.logging.set_verbosity(tf.logging.INFO)
    input_fn = lambda params : mnist_ds('train[0%:80%]', BATCH_SIZE)
    estimator.train(input_fn=input_fn, max_steps=TRAINING_STEPS)


def evaluate(img_labels_fn, steps=None):
    estimator = tf.keras.estimator.model_to_estimator(
            keras_model=compile_model(), model_dir=MODEL_DIR)
    results = estimator.evaluate(img_labels_fn, steps=steps)
    accuracy = results['acc']
    return accuracy


if __name__ == '__main__':
    train()
    exit(0)
    sample = tf.data.make_one_shot_iterator(
            mnist_ds('test', BATCH_SIZE)).get_next()
    with tf.Session() as sess:
        sample = sess.run(sample)
    # Sample is a tuple ([batch, img], [batch, label])
    # res = evaluate(lambda : sample, steps=sample[0].shape[0])
    res = evaluate(lambda : mnist_ds('test', BATCH_SIZE), 
                   steps=1024)
    print(res)
