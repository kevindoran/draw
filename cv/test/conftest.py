import tensorflow as tf
import pytest

@pytest.fixture
def tf_session():
    """Yeild a Tensorflow session. 

    This is needed so that multiple tests can easily create
    separate graphs without accedently sharing the same graph."""
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            yield sess

