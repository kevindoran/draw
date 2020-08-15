import tensorflow as tf
import tensorflow_datasets as tfds
import draw.vanilla_draw as vanilla_draw
import draw.mnist as mnist
import numpy as np
import pytest


def test_gru_init():
    # Setup
    input_len = 4
    hidden_len = 7
    # Test
    # No errors should be thrown:
    gru = vanilla_draw.Gru(input_len, hidden_len)


def test_gru_forward():
    # Setup
    input_len = 4
    hidden_len = 7
    gru = vanilla_draw.Gru(input_len, hidden_len)
    test_input = np.random.randn(input_len)
    initial_state = np.random.randn(hidden_len)
    # Test 
    # No errors should be thrown:
    output = vanilla_draw.gru_forward(gru, test_input, initial_state)


def test_draw_init():
    # Setup
    img_shape = (28, 28)
    encode_hidden_len = decode_hidden_len = 256
    latent_len = 15
    num_loops = 4
    # Test
    # No errors should be thrown:
    net = vanilla_draw.Draw(
        img_shape, num_loops, encode_hidden_len, latent_len, decode_hidden_len)


def test_draw_forward():
    # Setup
    img_shape = (28, 28)
    encode_hidden_len = decode_hidden_len = 256
    latent_len = 15
    num_loops = 10
    net = vanilla_draw.Draw(
        img_shape, num_loops, encode_hidden_len, latent_len, decode_hidden_len)
    test_img = np.random.randn(28 * 28) 
    prev_img = None
    h_enc = np.random.randn(encode_hidden_len)
    h_dec = np.random.randn(decode_hidden_len)
    # Test
    # No errors should be thrown:
    canvas, _ = vanilla_draw.draw_forward(net, test_img)


def test_gru_forward_and_back():
    """Test a back-propagation on a very small GRU unit.

    GRU
    ===
    Inputs and weights
    ------------------
    x = [0.02, -0.3]
    h_prev = [0]
    xh_prev = [0.02, -0.3, 0]
    # Wr and Wz are shape [1, 3], [hidden_len, input_len + hidden_len]
    Wr = [[0.5, 0.5, 0.1]]
    Wz = [[0.5, 0.5, 0.1]]
    Wp = [[0.5, 0.5, 0.1]]
    br = [0.5]
    bz = [0.5]
    bp = [0.5]

    Forward computation
    -------------------
    (1) r_linear = Wr[x, h_prev] + br
    (2) r = sigmoid(r_linear)
    (3) z_linear = Wz[x, h_prev] + bz
    (4) z = sigmoid(z_linear)
    (5) h_reset = r.h_prev  
    (6) p_linear = Wr[x, h_reset] + bp
    (7) p = tanh(p_linear)
    (8) h = (1 - z).p + z.h_prev

    (1) r_linear = [[0.5, 0.5, 0.1]] @ [0.02, -0.3, 0] + [0.5] = [0.36]
    (2) r = sigmoid([0.36]) = [0.58904]
    (3) z_linear = [[0.5, 0.5, 0.1]] @ [0.02, -0.3, 0] + [0.5] = [0.36]
    (4) z = sigmoid([0.36]) = [0.58904]
    (5) h_reset = [0.46506] * [0] = [0]
        xh_reset = [0.02, -0.3, 0]
    (6) p_linear = [[0.5, 0.5, 0.1]] @ [0.02, -0.3, 0] + [0.5] = [0.36]
    (7) p = tanh([0.36]) = [0.34521]
    (8) h = (1 - [0.58904]) * [0.34521] + [0.58904]*[0] = 0.79665
    """

     # Setup
    hidden_len = 1
    input_len = 2
    weight_init_val = 0.5
    x = np.array([0.02, -0.3])
    dh = np.array([2.1])
    init_function = lambda shape: np.full(shape, weight_init_val)
    zero_init = lambda shape : np.zeros(shape)
    gru = vanilla_draw.Gru(input_len, hidden_len, init_fctn=init_function)
    # Don't have the matrices all the same value, otherwise some bugs might
    # slip past.
    gru.Wr[(0, 2)] = gru.Wz[(0, 2)] = gru.Wp[(0, 2)] = 0.1
    dgru = vanilla_draw.Gru(input_len, hidden_len, init_fctn=zero_init)
    h, activations = vanilla_draw.gru_forward(gru, x, h=np.zeros(hidden_len))

    # Forward
    # Check activations against hand calculations.
    assert np.isclose(activations.r[0], 0.589040434057), 'wrong r activation.'
    assert np.isclose(activations.z[0], 0.589040434057), 'wrong z activation.'
    assert np.isclose(activations.p[0], 0.345214034135), 'wrong p activation.'
    assert np.isclose(activations.h[0], 0.141869009625), 'wrong h activation.'

    # Back-propagation
    """

    (8) h = (1 - z).p + z.h_prev
    dz = dh * (-p + h_prev) 
       = [2.1] * (-[0.34512] + [0]) 
       = [-0.72495] 
    dp = dh * (1 - z)
       = [2.1] * ([1] - [0.58904])
       = [0.86302]

    (3) z_linear = Wz[x, h_prev] + bz
    (4) z = sigmoid(z_linear)
    dz_linear = dz * z*(1 - z) 
              = [-0.72495] * [0.24207] 
              = [-0.17549]

    (6) p_linear = Wr[x, h_reset] + bp
    (7) p = tanh(p_linear)
    dp_linear = dp * (1 - p**2)
              = [0.86302] * ([1] - [0.34521]**2)
              = [0.76017]
    dxh_reset = transpose(Wp) @ dp_linear
             = [[0.5], [0.5], [0.1]] @ [0.76017] 
             = [[0.38008], [0.38008], [0.076017]]
    dh_reset = dxh_reset[input_len:] = [0.076017]

    (5) h_reset = r.h_prev  
    dr = dh_reset * h_prev
       = [0.076017] * [0]
       = [0]

    (1) r_linear = Wr[x, h_prev] + br
    (2) r = sigmoid(r_linear)
    dr_linear = dr * r /(1 - r)
              = [0] * ...
              = [0]

    # Collect dx h_prev from (8), (5), (3) and (1).
    dxh_prev = transpose(Wr) @ dr_linear
             = [[0.5], [0.5], [0.1]] @ [0]
             = [0, 0, 0]
    dxh_prev = transpose(Wz) @ dz_linear 
             = [[0.5], [0.5], [0.1]] @ [-0.17549] 
             = [[-0.087745], [-0.087745], [-0.017549]]
    dx = [0, 0] + [-0.087745, -0.087745] + dxh_reset[:input_len]
       = [0, 0] + [-0.087745, -0.087745] + [0.38008, 0.38008]
       = [0.29234, 0.29234]
    dh_prev = (dh * z) + [0] + [-0.017549] + (dxh_reset[input_len:] * r)
            = [2.1] * [0.58904] + [0] + [-0.017549] + [0.044777]
            = [1.26421] 
    """
    dx, dh_prev = vanilla_draw.backprop_gru(dh, gru, dgru, activations)
    assert np.isclose(dh_prev, 1.26421), 'h_prev gradient signal is wrong.'
    assert np.allclose(dx, [0.29234, 0.29234]), 'x gradient signal is wrong.'

    # Finally, let's check the updates to the weights.
    """
    dWz = dz_linear @ xh_prev 
        = [-0.17549] @ [0.02, -0.3, 0]
        = [[-0.0035098, 0.052647, 0]
    dWr = dr_linear @ xh_prev
        = [0] @ [0.02, -0.3, 0]
        = [[0, 0, 0]]
    dWp = dp_linear @ xh_reset
        = [0.76017] @ [0.02, -0.3, 0]
        = [[0.015203, -0.22805, 0]]
    dbz = dz_linear
        = [-0.17549]
    dbr = dr_linear
        = [0]
    dbp = dp_linear
        = [0.76017]
    """
    assert np.allclose(dgru.Wz, [[-0.0035098, 0.052647, 0.]]), \
            'Gradient of Wz is wrong.'
    assert np.allclose(dgru.Wr, [[0., 0., 0.]]), 'Gradient of Wr is wrong.'
    assert np.allclose(dgru.Wp, [[0.0152033, -0.22805, 0.]]), \
            'Gradient of Wp is wrong.'
    assert np.isclose(dgru.bz, -0.17549), 'Gradient of bz is wrong.'
    assert np.isclose(dgru.br, 0.), 'Gradient of br is wrong.'
    assert np.isclose(dgru.bp, 0.76017), 'Gradient of bp is wrong.'


def test_train():
    # No errors should be thrown:
    vanilla_draw.train(num_loops=10, learning_rate=1e-5, steps=20)


def test_sample(tf_session):
    np.seterr(all='raise')
    num_loops = 3
    steps = 128
    ds = tfds.as_numpy(mnist.mnist_ds('train', batch_size=1))
    generated_imgs = []
    labels = []
    # TODO: work to restore from file.
    draw_model = vanilla_draw.train(num_loops=num_loops, final_learning_rate=1e-8, 
            steps=100000)
    for s in range(steps):
        (img_in, label) = next(ds)
        # Remove batch dimension.
        img_in = img_in[0]
        img_out, _ = vanilla_draw.draw_forward(draw_model, img_in)
        import cv2 as cv
        cv.imwrite(f'./out/img_{s}_label_{label[0]}.png', img_out*255)
        # Add the batch dimension back in (needed for mnist evaluator).
        img_out = np.expand_dims(img_out, axis=0)
        # Also, Tensorflow accepts float32, and our numpy arrays have been
        # float64 (is this a default for numpy?).
        img_out = np.float32(img_out)
        generated_imgs.append(img_out)
        labels.append(label)
    input_fn = tf.estimator.inputs.numpy_input_fn(
            x=np.concatenate(generated_imgs), y=np.concatenate(labels), 
            shuffle=False)
    accuracy = mnist.evaluate(input_fn)
    ACCURACY_THRESHOLD = 0.70
    assert accuracy > ACCURACY_THRESHOLD, ('Accuracy a bit low. Possibly a '
            'regression has occurred.')
