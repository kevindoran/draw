import tensorflow as tf
from collections import namedtuple
import cv2 as cv
import tensorflow_datasets as tfds
import math
import draw.mnist
import pathlib

CHECKPOINT_PATH = './out/models/tf_draw/'
BATCH_SIZE = 128

def model(x_in, enc_size, dec_size, z_size, num_loops, batch_size):
    x_in = x_in * 1/255
    x_in = tf.cast(x_in, tf.float32)
    #x_in = tf.placeholder(tf.float32,shape=(batch_size,img_shape)) 
    assert len(x_in.shape) == 3, "Expecting (batch, height, width)."
    # Flatten the input to be 1D
    x_in_flat = tf.reshape(x_in, [-1, x_in.shape[1]*x_in.shape[2]])
    img_shape = x_in_flat.shape[1:]
    rand = tf.random.normal((batch_size,z_size), mean=0, stddev=1)
    enc_rnn = tf.keras.layers.GRUCell(enc_size, name='enc_GRU')
    dec_rnn = tf.keras.layers.GRUCell(dec_size, name='dec_GRU')
    mean_dense = tf.keras.layers.Dense(1, name='mean_dense')
    log_sd_dense = tf.keras.layers.Dense(1, name='log_sd_dense')
    write_dense = tf.keras.layers.Dense(*img_shape, name='write_dense')

    def sampleZ(h_enc):
        mean = mean_dense(h_enc)
        log_sd = log_sd_dense(h_enc)
        sd = tf.keras.activations.exponential(log_sd)
        return (mean + sd*rand, mean, log_sd, sd)

    def read(x, x_hat):
        return tf.concat([x, x_hat], 1)

    def write(h_dec):
        return write_dense(h_dec)

    LoopState = namedtuple('LoopState', 
            ['c', 'h_enc', 'h_dec', 'z', 'mean', 'sd', 'log_sd'])
    
    def single_loop(x, c_prev, h_enc_prev, h_dec_prev):
        x_hat = x - tf.sigmoid(c_prev)
        # Read
        r  = read(x, x_hat)
        # Encode
        # Return type for GRUCell is h,[h]. First h is output, the second
        # [h] is a list of internal states to pass to the next timestep.
        h_enc = enc_rnn(tf.concat([r, h_dec_prev], 1), [h_enc_prev])[0]
        # Sample
        z, mean, log_sd, sd = sampleZ(h_enc)
        # Decode
        h_dec = dec_rnn(z, [h_dec_prev])[0]
        # Write
        c = c_prev + write(h_dec)
        return LoopState(c, h_enc, h_dec, z, mean, sd, log_sd)

    def _loss(loop_states, x_in, x_out):
        # Reconstruction loss
        def binary_cross_entropy(target, out):
            # For numerical stability.
            epsilon = 1e-8 
            return -(target * tf.math.log(out + epsilon) + 
                     (1.0 - target) * tf.math.log(1.0 - out + epsilon))
        # lx = tf.reduce_sum(binary_cross_entropy(x_in, x_out), 1)
        # Why is this separated into reduce sum and reduce mean?
        # lx = tf.reduce_mean(lx)
        lx = tf.reduce_mean(binary_cross_entropy(x_in, x_out))

        # KL divergence
        kl_terms = []
        for s in loop_states:
            mean_sq = tf.square(s.mean)
            sd_sq = tf.square(s.sd)
            kl_terms.append(
                    0.5 * tf.reduce_sum(mean_sq + sd_sq - 2 * s.log_sd, 1) 
                    - 0.5)
                    #- len(loop_states) * 0.5)
        kl_sum = tf.add_n(kl_terms)
        lz = tf.reduce_mean(kl_sum)
        loss = lx + 0.1 * lz
        return loss

    x = x_in_flat
    c_prev = tf.zeros((batch_size, *img_shape))
    h_enc_prev = tf.zeros((batch_size, enc_size))
    h_dec_prev = tf.zeros((batch_size, dec_size))
    loop_states = []
    with tf.variable_scope('loop', reuse=tf.AUTO_REUSE):
        for l in range(num_loops):
            state = single_loop(x, c_prev, h_enc_prev, h_dec_prev)
            loop_states.append(state) 
            c_prev = state.c
            # Interesting: forgetting to feed the state back into the encoder
            # by skipping the next line causes no noticeable degradation. 
            h_enc_prev = state.h_enc
            h_dec_prev = state.h_enc
    x_out = tf.math.sigmoid(c_prev)
    loss_op = _loss(loop_states, x, x_out)
    # Reshape back to 2D and rescale.
    x_out = x_out * 255
    x_out = tf.reshape(x_out, [-1, x_in.shape[1], x_in.shape[2]])
    return x_out, loss_op


#def create_dataset_iterator():
#    def create_shape_ds():
#        return data.BlackWhiteShapeDataset1(img_shape=[28,28])

def create_model(x_in):
    x_out, loss = model(x_in, enc_size=256, dec_size=256, z_size=10, 
                        num_loops=10, batch_size=BATCH_SIZE)
    return (x_out, loss)
    

def train():
    train_steps = 5000
    ds = draw.mnist.mnist_ds('test', batch_size=BATCH_SIZE)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    img_input = iterator.get_next()[0]
    x_in = img_input
    x_out, loss = create_model(x_in)
    for v in tf.global_variables():
        print(f'{v.name}: {v.get_shape()}')
    learning_rate = 1.0e-5
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    grads = optimizer.compute_gradients(loss)
    # I was getting NaNs. Worth investigating further.
    for i,(g,v) in enumerate(grads):
        if g is not None:
            grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op = optimizer.apply_gradients(grads)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(train_steps):
            (_, loss_val) =sess.run([train_op, loss])
            if math.isnan(loss_val):
                print(f'Step: {t}. Loss: NaN')
                break
            else:
                print(f'Step: {t}. Loss: {float(loss_val)}')
            if t > 0 and t % 500 == 0:
                (img_in, img_out) = sess.run([img_input, x_out])
                num_print = 2
                for i in range(num_print):
                    cv.imwrite(f'./out/image_input_step_{t}_{i}.png', img_in[i])
                    cv.imwrite(f'./out/image_output_step_{t}_{i}.png', img_out[i])
    
        # Save graph
        saver = tf.train.Saver()
        pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
        saver.save(sess, CHECKPOINT_PATH, global_step=t)


def generate(input_ds):
    iterator = tf.compat.v1.data.make_one_shot_iterator(input_ds)
    img_input = iterator.get_next()[0]
    x_in = img_input
    img_out, loss = create_model(x_in)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_PATH))
        import pdb; pdb.set_trace();
        img_in_val, img_out_val = sess.run([img_input, img_out])


def generate_test():
    ds = draw.mnist.mnist_ds('test', batch_size=BATCH_SIZE)
    generate(ds)


if __name__ == '__main__':
    # train()
    generate_test()
       

        #for idx, img in enumerate(img_input):
            #cv.imwrite(f'./out/image_input_{idx}.png', img)
