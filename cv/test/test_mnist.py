import draw.mnist
import tensorflow as tf

def test_evaluate():
    """Tests that the MNIST recogition model is properly trained and can
    categorize MNIST digits with high accuracy."""
    img_labels = draw.mnist.mnist_ds('test', batch_size=128)
    acc = draw.mnist.evaluate(
            lambda : draw.mnist.mnist_ds('test', batch_size=128), 
            steps=1024)
    accuracy_threshold = 0.95
    assert acc > accuracy_threshold, 'Accuracy should be at least {accuracy_threshold}.'


    
