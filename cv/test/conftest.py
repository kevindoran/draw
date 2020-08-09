import tensorflow as tf
import pytest

@pytest.fixture
def tf_session():
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            yield sess

