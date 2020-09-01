import numpy as np
import tensorflow as tf
import draw.vanilla_draw as vanilla_draw
import draw.tf_draw as tf_draw


def _trainable_variable_values(sess):
    trainables = tf.trainable_variables()
    values = sess.run(trainables)
    names = (v.name for v in trainables)
    return dict(zip(names, values))

def _get_tensor(name):
    return tf.get_default_graph().get_tensor_by_name(name)


def populate_GRU(sess, vanilla_gru, tf_gru_cell_name):
    [tf_rz_kernel, tf_rz_bias, tf_p_kernel, tf_p_bias] = sess.run([
            _get_tensor(f'{tf_gru_cell_name}/gates/kernel:0'),
            _get_tensor(f'{tf_gru_cell_name}/gates/bias:0'),
            _get_tensor(f'{tf_gru_cell_name}/candidate/kernel:0'),
            _get_tensor(f'{tf_gru_cell_name}/candidate/bias:0')
        ])
    # Transpose, as Tensorflow effectively runs transpose(x).transpose(W).
    vanilla_gru.Wr = np.transpose(np.hsplit(tf_rz_kernel, 2)[0])
    vanilla_gru.Wz = np.transpose(np.hsplit(tf_rz_kernel, 2)[1])
    vanilla_gru.br = np.hsplit(tf_rz_bias, 2)[0]
    vanilla_gru.bz = np.hsplit(tf_rz_bias, 2)[1]
    vanilla_gru.Wp = np.transpose(tf_p_kernel)
    vanilla_gru.bp = tf_p_bias 
   

def get_random_vals(vanilla_draw, sess):
    random_values = []
    for l in range(vanilla_draw.num_loops):
        random_values.append(_get_tensor(f'loop/sample_random_e_{l}:0'))
    return random_values


def use_tf_weights(vanilla_draw, sess):
    populate_GRU(sess, vanilla_draw.enc_rnn, 'loop/enc_GRU')
    populate_GRU(sess, vanilla_draw.dec_rnn, 'loop/dec_GRU')
    # init enc_h and dec_h?
    # For some reason, the below tensors are being returned as resource
    # types, which I'm not sure how to get the actual value from.
    # W_μ, b_μ, W_σ, b_σ, W_write, b_write = sess.run([
    #     _get_tensor('loop/mean_dense/kernel:0'),
    #     _get_tensor('loop/mean_dense/bias:0'),
    #     _get_tensor('loop/log_sd_dense/kernel:0'),
    #     _get_tensor('loop/log_sd_dense/bias:0'),
    #     _get_tensor('loop/write_dense/kernel:0'),
    #     _get_tensor('loop/write_dense/bias:0')
    #     ])
    # vanilla_draw.W_μ = W_μ 
    # vanilla_draw.b_μ = b_μ
    # vanilla_draw.W_σ = W_σ
    # vanilla_draw.b_σ = b_σ
    # vanilla_draw.W_write = W_write 
    # vanilla_draw.b_write = b_write

    trainables = _trainable_variable_values(sess)
    vanilla_draw.W_μ = np.transpose(trainables['loop/mean_dense/kernel:0'])
    vanilla_draw.b_μ = trainables['loop/mean_dense/bias:0']
    vanilla_draw.W_σ = np.transpose(trainables['loop/log_sd_dense/kernel:0'])
    vanilla_draw.b_σ = trainables['loop/log_sd_dense/bias:0']
    vanilla_draw.W_write = np.transpose(trainables['loop/write_dense/kernel:0'])
    vanilla_draw.b_write = trainables['loop/write_dense/bias:0']

