# Our only import.
import numpy as np
import logging

# Used to avoid division by zero in some places. 
_epsilon = 1e-8 

def sigmoid(x):
    return  1 / (1 + np.exp(-x))


def d_sigmoid(y):
    """Computes dy/dx of y=sigmoid(x), in terms of y."""
    return y * (1 - y)


def d_tanh(y):
    """Computes dy/dx of y=tanh(x), in terms of y."""
    return 1 - y**2


def d_exp(y):
    """Computes dy/dx of y=exp(x), in terms of y.
    
    By definition, the result is simply y. I keep this method even though it
    is so simple, as without it, it is not clear that a derivative operation
    has taken place.
    """
    return y


def binary_cross_entropy(target, out):
    """Computes the binary cross entropy of target with respect to out. 

    Args:
        target: array of bernoulli distribution parameters.
        out: array of bernoulli distribution parameters.
    """
    return -(target * np.log(out + _epsilon) + 
             (1.0 - target) * np.log(1.0 - out + _epsilon))


def d_binary_cross_entropy(target, out):
    """Computes ∂y/∂o of y(t, o) = binary_cross_entropy(t, o)."""
    return -(target / (out + _epsilon)) + (1 - target) / (1 - out + _epsilon)


class Gru:
    """Encapsulates the state of a GRU (gated recurrent unit).

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
        self.input_len = input_len
        self.hidden_len = hidden_len
        if init_fctn is None:
            init_fctn = lambda shape : 0.1 * np.random.randn(*shape) - 0.05
        self.Wz = init_fctn([hidden_len, input_len + hidden_len])
        self.Wr = init_fctn([hidden_len, input_len + hidden_len])
        self.Wp = init_fctn([hidden_len, input_len + hidden_len])
        self.bz = init_fctn([hidden_len])
        # The reset and weighting signals need special initialization.
        if init_fctn is None:
            self.br = np.fill(shape, -1.0)
            self.bz = np.fill(shape, 0.5)
        else:
            self.br = init_fctn([hidden_len])
            self.bz = init_fctn([hidden_len])
        self.bp = init_fctn([hidden_len])

    def update(self, other, factor):
        """Add to the GRU's weights (this is used when training)."""  
        self.Wz += other.Wz * factor
        self.Wr += other.Wr * factor 
        self.Wp += other.Wp * factor 
        self.bz += other.bz * factor 
        self.br += other.br * factor 
        self.bp += other.bp * factor 


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
        """
        Args:
            c_prev: the previous canvas.
            enc (GruActivation): the activation for the encoder GRU.
            Q (QActivation): the activation for the sampler tensors.
            dec (GruActivation): the activation for the decoder GRU.
            c: the canvas output
        """
        self.c_prev = c_prev
        self.enc = enc
        self.dec = dec
        self.Q = Q
        self.c = c


def gru_forward(gru, x, h):
    """Compute the output of the GRU."""
    z = sigmoid(gru.Wz @ np.concatenate([x, h]) + gru.bz)
    r = sigmoid(gru.Wr @ np.concatenate([x, h]) + gru.br)
    p = np.tanh(gru.Wp @ np.concatenate([x, r * h]) + gru.bp)
    h_next = (1 - z) * p + z * h
    # Training requires the activations. They are returned here, but this
    # would normally not be none when not training.
    activations = GruActivation(x, h, r, z, p, h_next)
    return h_next, activations


class Draw:
    """Encapsulates the state of a DRAW network."""
    def __init__(self, img_shape, num_loops, encode_hidden_len, latent_len, 
            decode_hidden_len, init_fctn=None):
        """Construct a Draw network.

        Args:
            img_shape (np.narray): shape of the input and generated images.
        """
        self.img_shape = img_shape
        self.img_len = np.prod(img_shape)
        self.num_loops = num_loops
        self.latent_len = latent_len
        enc_input_len = self.img_len * 2
        # The initialization function both:
        #   * has a huge effect on training effectiveness.
        #   * has a huge effect on the tendency to encouter an
        #     underflow/overflow at the beginning of training.
        if init_fctn is None:
            init_fctn = lambda shape : 0.1 * np.random.randn(*shape) # - 0.03
            self.enc_rnn = Gru(enc_input_len, encode_hidden_len)
            self.dec_rnn = Gru(latent_len, decode_hidden_len)
        else:
            self.enc_rnn = Gru(enc_input_len, encode_hidden_len, init_fctn)
            self.dec_rnn = Gru(latent_len, decode_hidden_len, init_fctn)
        self.W_μ =     init_fctn([latent_len, encode_hidden_len])
        self.b_μ =     init_fctn([latent_len])
        self.W_σ =     init_fctn([latent_len, encode_hidden_len])
        self.b_σ =     init_fctn([latent_len])
        self.W_write = init_fctn([self.img_len, decode_hidden_len])
        self.b_write = init_fctn([self.img_len])

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


def _write(draw, dec_h, canvas):
    """Run the given DRAW network's write operation on a canvas."""
    return canvas + draw.W_write @ dec_h + draw.b_write


def _sample(draw, enc_h, e_override=None):
    """Sample from a DRAW networks generation distribution."""
    e = np.random.standard_normal(draw.latent_len) if e_override is None \
           else e_override
    μ = draw.W_μ @ enc_h + draw.b_μ
    log_σ = draw.W_σ @ enc_h + draw.b_σ
    σ = np.exp(log_σ) 
    z = (μ + σ * e)
    activations = QActivation(μ, σ, e, z)
    return activations


def _draw_forward_once(draw, img_in, enc_h, dec_h, c_prev, e_override=None):
    """Forward run of the DRAW network."""
    xx_hat = np.concatenate([img_in, img_in - sigmoid(c_prev)])
    enc_h, enc_act = gru_forward(draw.enc_rnn, xx_hat, enc_h)
    Q_act = _sample(draw, enc_h, e_override)
    dec_h, dec_act = gru_forward(draw.dec_rnn, Q_act.z, dec_h)
    c = _write(draw, dec_h, c_prev)
    activations = DrawActivation(c_prev, enc_act, Q_act, dec_act, c) 
    return activations


def draw_forward(draw, img_in, e_overrides=None):
    """Run all loops of a DRAW network for a given input image."""
    img_in = img_in.flatten()
    canvas = np.zeros(img_in.shape)
    enc_h = np.zeros(draw.enc_rnn.hidden_len)
    dec_h = np.zeros(draw.dec_rnn.hidden_len)
    activations = []
    for l in range(draw.num_loops):
        e = None if not e_overrides else e_overrides[l]
        act = _draw_forward_once(draw, img_in, enc_h, dec_h, canvas, e)
        activations.append(act)
        enc_h = act.enc.h
        dec_h = act.dec.h
        canvas = act.c
    canvas = sigmoid(canvas)
    res = np.reshape(canvas, draw.img_shape)
    return res, activations


# Note: assume that the error signal contains error from next step's hidden.
def backprop_gru(e, gru, dgru, gru_act):
    """Back-propagate error gradients through a GRU unit.

    The gradients for each trainable variable are computed and returned.

    Args:
        e: the change in loss/error with respect to the output. Note: it is
           required that this error signal contains the error from both the
           directly connected output and the error from the feedback loop.
        gru: the GRU unit for which to back-propagate through.
        dgru: gradients will added to dgru (incrementing existing values).
        gru_act: the activations of the given GRU unit.
    Returns:
        (dx, dh_prev): the change in loss wrt. the input signal and hidden 
                       state signal.
    """
    # Our outputs, dx and dh_prev. They will be incremented at multiple 
    # points within this method.
    dx = 0
    dh_prev = 0

    # Proceed in reverse. The equation numbers cross-reference the equation 
    # numbers listed in the Gru class above. Refer to the test file for a 
    # manual back-propagation walkthrough of these steps. 

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
    """Back-propagate the error signal through the sampling layer.
    
    The sampling layer deals with both the error signal coming from its output
    and the error signal generated as a result of the KL-divergence with respect
    to our prior/goal distribution (standard normal).

    This method isn't well tested, and is somehere I would consider not-unlikely
    to contain mistakes.
    """
    # Distribution loss
    # kl_loss = 1/2( draw_act.μ**2 + draw_act.σ**2 - 2*draw_act.log_σ)
    # If kl_factor is too high (1.0), then it's hard to avoid numeric errors.
    kl_factor = 0.1
    dμ = kl_factor * draw_act.Q.μ
    dσ = kl_factor * draw_act.Q.σ
    dσ_log = kl_factor * -1.0 * np.ones(draw_act.Q.σ.shape)

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


def backprop_draw(dc_linear, denc_h, ddec_h, draw, ddraw, draw_act):
    """Back-propagate error through one loop of a DRAW network.

    An important assumption for this method is that the dc_linear error given
    as input contains both the direct contribution to the cross-entropy loss
    and the loss that has propagated back from the next loop, if any.

    Args:
        dc_linear: the error gradient coming from both a) the cross-entropy loss
            and b) the error fedback from the next loop.
        denc_h: the error gradient from the encoder's hidden state connected 
            to the next layer, if any.
        ddec_h: the error gradient from the decoder's hidden state connected 
            to the next layer, if any.
        draw: the draw network.
        ddraw: the draw object where gradients will be stored.
        draw_act: all of the activations for this loop.
    Returns:
        (dc_prev, denc_h_prev, ddec_h_prev): 
           1) the loss gradient wrt. the loop's input.
           2) the loss gradient wrt. the encoder's hidden state.
           3) the loss gradient wrt. the decoder's hidden state.
    """
    ddraw.W_write = np.outer(dc_linear, draw_act.dec.h)
    ddraw.b_write = dc_linear 
    # This next line is subtle but important for the recursive behaviour. We
    # must *add* the two different errors to create ddec_h.
    ddec_h += np.transpose(draw.W_write) @ dc_linear
    dz, ddec_h_prev = backprop_gru(ddec_h, draw.dec_rnn, ddraw.dec_rnn, 
            draw_act.dec)
    # This next line is subtle but important for the recursive behaviour. We
    # must *add* the two different errors to create denc_h.
    denc_h += backprop_Q(dz, draw, ddraw, draw_act)
    dxx_hat, denc_h_prev = backprop_gru(denc_h, draw.enc_rnn, ddraw.enc_rnn,
            draw_act.enc)
    dx_hat = np.split(dxx_hat, 2)[1]
    dc_prev = -dx_hat * d_sigmoid(draw_act.c_prev)
    return dc_prev, denc_h_prev, ddec_h_prev


def dloss(draw, img_in):
    """Run back-propagation for a whole DRAW network (for all loops)."""
    init_fctn = lambda shape : np.zeros(shape)
    # Duplicate the set of trainable variables to store the gradients.
    ddraw = Draw(draw.img_shape, draw.num_loops, draw.enc_rnn.hidden_len, 
            draw.latent_len, draw.dec_rnn.hidden_len, init_fctn=init_fctn)
    # Run the whole network and record the output and all activations.
    img_out, activations =  draw_forward(draw, img_in)
    # Flatten the images, as our network take the data in as a vector.
    flat_img_in = img_in.flatten()
    flat_img_out = img_out.flatten()
    loss = binary_cross_entropy(flat_img_in, flat_img_out)
    dc = d_binary_cross_entropy(flat_img_in, flat_img_out)
    dc_linear = dc * d_sigmoid(flat_img_out)
    dc_prev = 0
    #ddraw.b_write = dc_linear
    denc_h_prev = np.zeros(draw.enc_rnn.hidden_len)
    ddec_h_prev = np.zeros(draw.dec_rnn.hidden_len)
    for i in reversed(range(draw.num_loops)):
        # This next line is critical in theory and practice! 
        dc_sum = dc_prev + dc_linear
        dc_prev, denc_h_prev, ddec_h_prev = backprop_draw(dc_sum, 
                denc_h_prev, ddec_h_prev, draw, ddraw, activations[i])
        
    return np.sum(loss), ddraw


def sgd(draw, img_in, learning_rate):
    """Stochastic gradient descent.

    An extremely basic training step (no batching nor momentum etc.).

    Run the draw network for all of its loops and calculate loss gradients for
    the whole network. Then multiply weights by the gradients (with a factor).

    Returns:
        the binary cross-entropy loss of our output wrt to the input image.
    """
    loss, ddraw = dloss(draw, img_in)
    draw.update(ddraw, -learning_rate)
    return loss


def train(num_loops, final_learning_rate, steps):
    """Construct and train a draw network.

    This is the top level training function. It is the site of a lot of 
    hacky experimentation with training schedules, so it never looks tidy.
    """
    # Construct a Draw object with preset settings.
    enc_hidden_len = 256
    dec_hidden_len = 256
    latent_len = 10
    img_shape = [28, 28]
    draw = Draw(img_shape, num_loops, enc_hidden_len, latent_len, 
            dec_hidden_len) 

    # We are importing Tensorflow, but only for loading MNIST.
    import draw.mnist as mnist
    import tensorflow_datasets as tfds
    ds = tfds.as_numpy(mnist.mnist_ds('train', batch_size=1))
    logging.info('Beginning training.')
    log_every = 200
    warmup_steps = steps * 0.25
    # We want to start with a high learning rate, then decrease. Initially,
    # the network is very unstable and prone to numeric overflow/underflows,
    # so we start 'carefully'. After the careful period, we bump the learning
    # rate up to its maximum, then descend until we reach the given lower 
    # bound (final) learning rate.
    careful_boost = 500
    full_boost = 8000
    lr = final_learning_rate  * careful_boost
    for s in range(steps):
        if s == warmup_steps:
            lr = final_learning_rate * full_boost
        if s > warmup_steps and lr > final_learning_rate:
            lr  = lr * 0.995
        # Get our next image.
        (img_in, label) = next(ds)
        # Remove the batch dimension, then flatten.
        img_flat = img_in[0].flatten()
        # Run a step of stochastic gradient descent.
        loss = sgd(draw, img_flat, lr)
        # Periodically log.
        if s % log_every == 0:
            logging.info(f'Step: {s}/{steps},\t\tloss:{loss}')
    return draw


def sample(draw, img_in, num_loops):
    img_out = np.zeros(img_in.shape)
    for l in range(num_loops):
        img_out = draw.forward(img_in, img_out)
    img_out = np.reshape(img_out, [28, 28])
    return img_out
            

if __name__ == '__main__':
    train(num_loops=10, final_learning_rate=1e-5, steps=20000)






