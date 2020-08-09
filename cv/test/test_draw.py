import draw.mnist
import draw.tf_draw
import tensorflow as tf

def test_tf_draw():
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
    with tf.Session() as sess:
        generated_img, _ = draw.tf_draw.create_model(img)
        draw.tf_draw.restore_weights(sess)
        for s in range(steps):
            img_val, label_val, generated_img_val \
                    = sess.run([img, label, generated_img])
            generated_imgs.append(generated_img_val)
            label_val.append(label_val)
    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=np.asarray(generated_imgs), y=labels)
    accuracy = draw.mnist.evaluate(input_fn)
    import pdb;pdb.set_trace();
    print(accuracy)




