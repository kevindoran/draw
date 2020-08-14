# Our only import.
import numpy as np
import logging

_epsilon = 1e-8 

def sigmoid(x):
    return  1 / (1 + np.exp(-x))


def d_sigmoid(y):
    return y * (1 - y)


def d_tanh(y):
    return 1 - y**2


def d_exp(y):
    return y


def binary_cross_entropy(target, out):
    return -(target * np.log(out + _epsilon) + 
             (1.0 - target) * np.log(1.0 - out + _epsilon))


def d_binary_cross_entropy(target, out):
    return -(target / out) + (1 - target) / (1 - out)


def kl_loss_wrt_std_normal(μ, σ):
    # This comes directly from the paper without derivation, eq 11.   
    pass


def d_kl_loss_wrt_std_normal():
    # TODO
    pass


# Model parameters
class Gru:
    """Encapsulates the state and behaviour of a GRU (gated recurrent unit).

    Steps carried out within a GRU network:

    (1) z = σ(Wz[x_t, h_{t-1}] + bz)
    (2) r = σ(Wr[x_t, h_{t-1}] + br)
    (3) p = tanh(Wp[x_t, r.h_{t-1}] + bp)
    (4) h_t = (1 - z).p + z.h_{t-1}

    The GRU sub-network has two inputs and one output:

        h_t = Gru(x, h_{t-1})

    The network can be best understood in reverse, starting with the output
    h_t:

    h_t, our new state 
    ------------------
    h_t is a vector that is both the network output and the state to be fed 
    back into the GRU if it is to be run again.

    p, candidate state
    ------------------
    p is a vector that we are proposing to be the new state, h_t.  The new 
    state, h_t, will be calculated as a weighted update between the existing 
    state and the candidate state. This happens in step (4). p is calculated
    in step (3). The tanh activation means p is a vector with elements from 
    [-1, 1]. 

    z, update gate
    --------------
    z is the vector that acts as the weighting between between our candidate 
    state, p, and the existing state, h_t. z is calculated in step (1) and 
    used in step (4).

    r, reset gate
    -------------
    r is used to drop information from the existing state when determining the
    candidate state. It is calculated in step (2) and used in step (3).

    From steps (1)-(4), we see that there are a total of 3 trainable matricies
    and 3 trainable bias vectors:

        Matricies: Wz, Wr, Wp
        Bias vectors: bz, br, bp

    Optimizations
    -------------
    Tensorflow (and possibly other frameworks) implement an optimization that
    improves performance at the expense of making the behaviour less obvious.
    The matrices Wz and Wr are combined into one; z and r are combined in one
    vector which will be split apart. The computation looks like:

    (0) temp = σ(Wg[x_t, h_{t-1}] + bg)
    (1) z = first_half(temp)
    (2) r = second_half(temp)
    (3) p = tanh(Wp[x_t, r.h_{t-1}] + bp)
    (4) h_t = (1 - z).p + z.h_{t-1}

    With this optimization, there are only two trainable matrices Wg and Wp
    and two trainable biases, bg and bp. I think the optimization is carried
    out so as to reduce the number of matrix multiplications.

    """
    def __init__(self, input_len, hidden_len, init_fctn=None):
        if init_fctn is None:
            init_fctn = lambda shape : 0.01 * np.random.randn(shape)
        self.Wz = init_fctn([hidden_len, input_len + hidden_len])
        self.Wr = init_fctn([hidden_len, input_len + hidden_len])
        self.Wp = init_fctn([hidden_len, input_len + hidden_len])
        self.bz = init_fctn([hidden_len])
        self.br = init_fctn([hidden_len])
        self.bp = init_fctn([hidden_len])

    @property
    def hidden_len(self):
        return np.shape(self.Wz)[0]

    @property
    def input_len(self):
        return np.shape(self.Wz)[1] - self.hidden_len

    def update(self, other, factor):
        """Add to the GRUs weights (this is used when training)."""  
        self.Wz += other.Wz * factor
        self.Wr += other.Wr * factor 
        self.Wp += other.Wp * factor 
        self.bz += other.bz * factor 
        self.br += other.br * factor 
        self.bp += other.bp * factor 


# Note: these basic classes could be replaced with named tuples.
class GruActivation:
    def __init__(self, x, h_prev, r, z, p, h):
        self.x = x
        self.h_prev = h_prev
        self.r = r
        self.z = z
        self.p = p
        self.h = h


class QActivation:
    def __init__(self, μ, σ, e, z):
        self.μ = μ
        self.σ = σ
        self.e = e
        self.z = z


class DrawActivation:
    def __init__(self, c_prev, enc, Q, dec, c):
        # self.x = None
        self.c_prev = c_prev
        self.enc = enc
        self.dec = dec
        self.Q = Q
        # self.c_u = c_u
        self.c = c


def gru_forward(gru, x, h):
    z = sigmoid(gru.Wz @ np.concatenate([x, h]) + gru.bz)
    r = sigmoid(gru.Wr @ np.concatenate([x, h]) + gru.br)
    p = np.tanh(gru.Wp @ np.concatenate([x, r * h]) + gru.bp)
    h_next = (1 - z) * p + z * h
    # Training requires the activations. They are returned here, but this
    # would normally not be none when not training.
    activations = GruActivation(x, h, r, z, p, h_next)
    return h_next, activations



class Draw:

    def __init__(self, img_shape, encode_hidden_len, latent_len, 
            decode_hidden_len, init_factor=0.01):
        """Construct a Draw network.

        Args:
            img_shape (np.narray): shape of the input and generated images.
        """
        self.img_shape = img_shape
        self.img_len = np.prod(img_shape)
        enc_input_len = self.img_len * 2
        self.enc_rnn = Gru(enc_input_len, encode_hidden_len, init_factor)
        self.dec_rnn = Gru(latent_len, encode_hidden_len, init_factor)
        # Optional: move the Gru hidden state vector into Gru class.
        #   * pros: encapsulate details relevant to GRU.
        #   * cons: slightly obscures how the network works.
        self.enc_h = np.zeros(encode_hidden_len)
        self.dec_h = np.zeros(decode_hidden_len)
        # Python 3 supports unicode variable names!
        self.W_μ =     init_factor * np.random.randn(latent_len, 
                decode_hidden_len)
        self.b_μ =     init_factor * np.random.randn(latent_len)
        self.W_σ =     init_factor * np.random.randn(latent_len, 
                decode_hidden_len)
        self.b_σ =     init_factor * np.random.randn(latent_len)
        self.W_write = init_factor * np.random.randn(self.img_len, 
                decode_hidden_len)
        self.b_write = init_factor * np.random.randn(self.img_len)

    @property
    def enc_hidden_len(self):
        return np.shape(self.enc_h)[0]

    @property
    def dec_hidden_len(self):
        return np.shape(self.dec_h)[0]

    @property
    def latent_len(self):
        return np.shape(self.W_μ)[0]

    def update(self, other, factor):
        """Add to the models weights (this is used when training)."""
        self.enc_rnn.update(other.enc_rnn, factor)
        self.dec_rnn.update(other.dec_rnn, factor)  
        self.W_μ += other.W_μ * factor
        self.W_σ += other.W_σ * factor
        self.b_μ += other.b_μ * factor
        self.b_σ += other.b_σ * factor
        self.W_write += other.W_write * factor
        self.b_write += other.b_write * factor

    # TODO: move these to be file level functions. 
    def _sample(self, e_override=None):
        e = np.random.standard_normal(self.latent_len) if e_override is None \
                else e_override
        μ = self.W_μ @ self.enc_h + self.b_μ
        log_σ = self.W_σ @ self.enc_h + self.b_σ
        σ = np.exp(log_σ)
        z = (μ + σ*e)
        activations = QActivation(μ, σ, e, z)
        return z, activations

    def _write(self, img):
        return img + self.W_write @ self.dec_h + self.b_write

    def forward(self, img_in, c_prev=None):
        """Forward run of the draw network."""
        if c_prev is None:
            c_prev = np.zeros(self.img_len)
        xx_hat = np.concatenate([img_in, img_in - c_prev])
        self.enc_h, enc_act = gru_forward(self.enc_rnn, xx_hat, self.enc_h)
        z, Q_act = self._sample()
        self.dec_h, dec_act = gru_forward(self.dec_rnn, z, self.dec_h)
        c = self._write(c_prev)
        activations = DrawActivation(c_prev, enc_act, Q_act, dec_act, c) 
        return c, activations

    def reset_hidden(self):
        self.enc_h = np.zeros(self.enc_h.shape[0])
        self.dec_h = np.zeros(self.dec_h.shape[0])


def draw_forward(draw, img_in, num_loops):
    img_in = img_in.flatten()
    canvas = None
    for l in range(num_loops):
        canvas, _ = draw.forward(img_in, canvas)
    canvas = sigmoid(canvas)
    res = np.reshape(canvas, draw.img_shape)
    return res


# Note: assume that the error signal contains error from next step's hidden.
def backprop_gru(e, gru, dgru, gru_act):
    # Our outputs dx and dh_prev start at zero.
    # Setting them at zero reduces the likelihood of introducing a bug by
    # moving around some of the satements below. dx and dh_prev should always 
    # be added to, not assigned.
    dx = 0
    dh_prev = 0
    # (4) h_t = (1 - z).p + z.h_{t-1}
    dz = e * (gru_act.h_prev - gru_act.p)
    dh_prev += e * gru_act.z
    dp = e * (1 - gru_act.z)

    # (3) p = tanh(Wp[x_t, r.h_{t-1}] + bp)
    dp_linear = dp * d_tanh(gru_act.p)
    xrh = np.concatenate([gru_act.x, gru_act.r * gru_act.h_prev])
    dgru.Wp += np.outer(dp_linear, xrh)
    dgru.bp += dp_linear
    d_xrh = np.transpose(gru.Wp) @ dp_linear
    # Undo the concatenation of x and the reset hidden input.
    dx += d_xrh[:gru.input_len]
    drh = d_xrh[gru.input_len:]
    dh_prev += drh * gru_act.r
    dr = drh * gru_act.h_prev
    
    # (2) r = σ(Wr[x_t, h_{t-1}] + br)
    dr_linear = dr * d_sigmoid(gru_act.r)
    xh = np.concatenate([gru_act.x, gru_act.h_prev])
    dgru.Wr += np.outer(dr_linear, xh)
    dgru.br += dr_linear
    d_xh = np.transpose(gru.Wr) @ dr_linear
    dx += d_xh[:gru.input_len]
    dh_prev += d_xh[gru.input_len:]

    # (1) z = σ(Wz[x_t, h_{t-1}] + bz)
    dz_linear = dz * d_sigmoid(gru_act.z)
    dgru.Wz += np.outer(dz_linear, xh)
    dgru.bz += dz_linear
    # Undo the concatenation of x and the hidden input.
    d_xh = np.transpose(gru.Wz) @ dz_linear
    dx += d_xh[:gru.input_len]
    dh_prev += d_xh[gru.input_len:]

    return dx, dh_prev


def backprop_Q(dz, draw, ddraw, draw_act): 
    # Distribution loss
    # kl_loss = draw_act.μ**2 + draw_act.σ**2 - draw_act.log_σ**2
    dσ = 2*draw_act.Q.σ
    dμ = 2*draw_act.Q.μ
    dσ_log = 2*np.log(draw_act.Q.σ) # Need to extract un log version?
    # Image loss
    # σ
    dσ += dz * draw_act.Q.e
    dσ_log += dσ * d_exp(draw_act.Q.σ)
    ddraw.W_σ += np.outer(dσ_log, draw_act.enc.h)
    ddraw.b_σ += dσ_log
    # μ
    dμ += dz
    ddraw.W_μ = np.outer(dμ, draw_act.enc.h)
    ddraw.b_μ = dμ
    # enc_h
    denc_h = np.transpose(draw.W_σ) @ dσ_log + np.transpose(draw.W_μ) @ dμ
    return denc_h


def backprop_draw(err, denc_h, ddec_h, draw, ddraw, draw_act):
    ddraw.W_write = np.outer(err, draw_act.dec.h)
    ddraw.b_write = err 
    # This next line is subtle but important for the recursive behaviour.
    ddec_h += np.transpose(draw.W_write) @ err
    dz, ddec_h_prev = backprop_gru(ddec_h, draw.dec_rnn, ddraw.dec_rnn, 
            draw_act.dec)
    denc_h += backprop_Q(dz, draw, ddraw, draw_act)
    dxx_hat, denc_h_prev = backprop_gru(denc_h, draw.enc_rnn, ddraw.enc_rnn,
            draw_act.enc)
    dx_hat = np.split(dxx_hat, 2)[1]
    dc_prev = -dx_hat
    return dc_prev, denc_h_prev, ddec_h_prev


         
def dloss(draw, img_in, num_loops):
    # Duplicate the set of trainable variables to store the gradients.
    ddraw = Draw(draw.img_shape, draw.enc_hidden_len, draw.latent_len, 
            draw.dec_hidden_len, init_factor=0)
    activations = []
    c_prev = None
    for i in range(num_loops):
        c_prev, act = draw.forward(img_in, c_prev)
        activations.append(act)
    output = sigmoid(c_prev)
    loss = binary_cross_entropy(img_in, output)
    dl = d_binary_cross_entropy(img_in, output)
    dc_prev = dl * d_sigmoid(output)
    ddraw.b_write = dc_prev
    denc_h_prev = np.zeros(draw.enc_hidden_len)
    ddec_h_prev = np.zeros(draw.dec_hidden_len)
    for i in reversed(range(num_loops)):
        dc_prev, denc_h_prev, ddec_h_prev = backprop_draw(dc_prev, denc_h_prev, 
            ddec_h_prev, draw, ddraw, activations[i])
        
    return np.sum(loss), ddraw


# note: maybe move num_loops into Draw class. It's sort-of sort-of not part of
# it.
def sgd(draw, img_in, num_loops, learning_rate):
    loss, ddraw = dloss(draw, img_in, num_loops)
    draw.update(ddraw, -learning_rate)
    return loss


def train(num_loops, learning_rate, steps):
    enc_hidden_len = 256
    dec_hidden_len = 256
    latent_len = 10
    img_shape = [28, 28]
    draw = Draw(img_shape, enc_hidden_len, latent_len, dec_hidden_len) 

    # We are importing Tensorflow for loading MNIST.
    import draw.mnist as mnist
    import tensorflow_datasets as tfds
    logging.info('Beginning training.')
    ds = tfds.as_numpy(mnist.mnist_ds('train', batch_size=1))
    # TODO: only using one image for testing of training; easy to stop
    # issues.
    # (img_in, label) = next(ds)
    for s in range(steps):
        (img_in, label) = next(ds)
        # Remove the batch dimension, then flatten.
        img_flat = img_in[0].flatten()
        draw.reset_hidden() # Keep?
        loss = sgd(draw, img_flat, num_loops, learning_rate)
        logging.info(f'Step: {s}/{steps},\t\tloss:{loss}')
    return draw


def sample(draw, img_in, num_loops):
    img_out = np.zeros(img_in.shape)
    for l in range(num_loops):
        img_out = draw.forward(img_in, img_out)
    img_out = np.reshape(img_out, [28, 28])
    return img_out
            


    







