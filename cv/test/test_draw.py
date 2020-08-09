import draw.mnist
import draw.tf_draw
import tensorflow as tf
import numpy as np

def test_tf_draw(tf_session):
    sess = tf_session
    batch_size = 128
    # Round number of samples to multiple of batch size.
    num_samples = (1024 // batch_size) * batch_size
    steps = int(num_samples / batch_size)
    assert num_samples == batch_size * steps, \
        'The batch size should be a divisor of the sample count.'
    # Ready the (image, label) input.
    ds = draw.mnist.mnist_ds('test', batch_size=batch_size)
    iterator = tf.data.make_one_shot_iterator(ds)
    img, label = iterator.get_next()
    # Generate samples from the autoencoder.
    generated_imgs = []
    labels = []
    generated_img, _ = draw.tf_draw.create_model(img)
    draw.tf_draw.restore_weights(sess)
    for s in range(steps):
        img_val, label_val, generated_img_val \
                = sess.run([img, label, generated_img])
        generated_imgs.append(generated_img_val)
        labels.append(label_val)
    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=np.concatenate(generated_imgs), y=np.concatenate(labels), shuffle=False)
    accuracy = draw.mnist.evaluate(input_fn)
    # After training a few times, 85% seems like a good lower bound. Below
    # this, a change might have triggered a regression.
    ACCURACY_THRESHOLD = 0.85
    assert accuracy > ACCURACY_THRESHOLD, ('Accuracy a bit low. Possibly a '
        'regression has occurred.')
    print(accuracy)




