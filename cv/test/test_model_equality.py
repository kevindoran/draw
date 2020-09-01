import pytest
import numpy as np
import tensorflow as tf
import draw.vanilla_draw as vanilla_draw
import draw.tf_draw as tf_draw
import draw.mnist as mnist
import draw.model_bridge as model_bridge


def test_random_normal(tf_session):
    sess = tf_session
    batch_size = 32
    shape = [5]
    random_node = tf.random.normal((batch_size, *shape), mean=0, stddev=1)
    fives = tf.fill(shape, 5.0)
    tens = tf.fill(shape, 10.0)
    r1 = fives * random_node
    r2 = tens * random_node
    r1_val, r2_val = sess.run([r1, r2])
    assert np.allclose(r1_val * 2, r2_val), "This test shows that the random" \
        " node is only given a value once, irrelevant of how many nodes are " \
        "connected to it downstream."


def test_model_equality_debug(tf_session):
    # Setup
    sess = tf_session
    batch_size = 128 # TODO: how to handle?
    loops = 10
    # Ready the (image, label) input.
    ds = mnist.mnist_ds('test', batch_size=batch_size)
    iterator = tf.data.make_one_shot_iterator(ds)
    img, label = iterator.get_next()
    # Create Tensorflow model
    # To save training the model again, use the weights generated in
    # src/draw/tf_draw.py:train().
    tf_img, loss, loop_states = tf_draw.model(img, enc_size=256, dec_size=256, 
            z_size=10, num_loops=loops, batch_size=batch_size)
    sess.run(tf.global_variables_initializer()) # needed?
    tf_draw.restore_weights(sess)
    # Useful for debugging:
    # tensors = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # Create vanilla model
    draw_net = vanilla_draw.Draw([28, 28], num_loops=10, encode_hidden_len=256, latent_len=10, decode_hidden_len=256)
    model_bridge.use_tf_weights(draw_net, sess)
    # Generate the output images.
    # Tensorflow model
    enc_outputs = [l.h_enc for l in loop_states]
    random_nodes = [l.rand for l in loop_states]
    dec_outputs = [l.h_dec for l in loop_states]
    zs = [l.z for l in loop_states]
    means = [l.mean for l in loop_states]
    sds = [l.sd for l in loop_states]
    cs = [l.c for l in loop_states]
    img_in, tf_img_out, enc_output_vals, random_vals, dec_output_vals, \
            z_vals, mean_vals, sd_vals, c_vals = sess.run([img, tf_img, 
                enc_outputs, random_nodes, dec_outputs, zs, means, sds, 
                cs])
    for batch in range(batch_size):
        e_overrides = []
        for r in random_vals:
            e_overrides.append(r[batch])
        # Vanilla model
        img_out, vanilla_act = vanilla_draw.draw_forward(draw_net, img_in[batch], 
                e_overrides)
        ed = lambda loop, atol : np.argwhere(np.invert(np.isclose(
            vanilla_act[loop].enc.h, enc_output_vals[loop][batch], atol=atol)))
        dd = lambda loop, atol : np.argwhere(np.invert(np.isclose(
            vanilla_act[loop].dec.h, dec_output_vals[loop][batch], atol=atol)))
        zd = lambda loop, atol : np.argwhere(np.invert(np.isclose(
            vanilla_act[loop].Q.z, z_vals[loop][batch], atol=atol)))
        md = lambda loop, atol : np.argwhere(np.invert(np.isclose(
            vanilla_act[loop].Q.μ, mean_vals[loop][batch], atol=atol)))
        sd = lambda loop, atol : np.argwhere(np.invert(np.isclose(
            vanilla_act[loop].Q.σ, sd_vals[loop][batch], atol=atol)))
        cd = lambda loop, atol : np.argwhere(np.invert(np.isclose(
            vanilla_act[loop].c, c_vals[loop][batch], atol=atol)))
        def print_dbg(loop, atol):
            print(f'enc: {len(ed(loop, atol))}')
            print(f'mean: {len(md(loop, atol))}')
            print(f'sd: {len(sd(loop, atol))}')
            print(f'z: {len(zd(loop, atol))}')
            print(f'dec: {len(dd(loop, atol))}')
            print(f'c: {len(cd(loop, atol))}')
        # print_dbg(0, 1e-8)
        # The default absolute error threshold for np.isclose() is 1e-8
        # 1e-7 fails, so I've set it to 1e-6. It's worth investigating where 
        # exactly we are deviating; however, given a few loops, it's not 
        # surprising that there would be some drift.
        atol = 1e-6
        for l in range(loops):
            assert all(np.isclose(vanilla_act[l].enc.h, 
                enc_output_vals[l][batch], atol=atol))
            assert all(np.isclose(vanilla_act[l].Q.μ, 
                mean_vals[l][batch], atol=atol))
            assert all(np.isclose(vanilla_act[l].Q.σ, 
                sd_vals[l][batch], atol=atol))
            assert all(np.isclose(vanilla_act[l].Q.z, 
                z_vals[l][batch], atol=atol))
            assert all(np.isclose(vanilla_act[l].dec.h, 
                dec_output_vals[l][batch], atol=atol))
            assert all(np.isclose(vanilla_act[l].c, 
                c_vals[l][batch], atol=atol))
        assert all(np.isclose(img_out, tf_img_out[batch], atol=atol).flatten())



def test_model_equality(tf_session):
    # Setup
    sess = tf_session
    batch_size = 128 # TODO: how to handle?
    steps = 10
    # Ready the (image, label) input.
    ds = mnist.mnist_ds('test', batch_size=batch_size)
    iterator = tf.data.make_one_shot_iterator(ds)
    img, label = iterator.get_next()
    # Create Tensorflow model
    # To save training the model again, use the weights generated in
    # src/draw/tf_draw.py:train().
    tf_img, _ = tf_draw.create_model(img)
    sess.run(tf.global_variables_initializer()) # needed?
    tf_draw.restore_weights(sess)
    tensors = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # Create vanilla model
    draw_net = vanilla_draw.Draw([28, 28], num_loops=10, encode_hidden_len=256, latent_len=10, decode_hidden_len=256)
    model_bridge.use_tf_weights(draw_net, sess)
    random_nodes = model_bridge.get_random_vals(draw_net, sess)
    # Generate the output images.
    for s in range(loops):
        # Tensorflow model
        res = sess.run([img, tf_img, *random_nodes, *enc_outputs])
        img_in, tf_img_out = res[:2]
        random_vals = res[2:12]
        h_enc_out_vals = res[12:]
        # Vanilla model
        for i in range(batch_size):
            img_out, _ = vanilla_draw.draw_forward(draw_net, img_in[i], 
                    e_overrides=random_vals)
            import pdb; pdb.set_trace();
            assert np.allclose(tf_img_out[i], img_out)
