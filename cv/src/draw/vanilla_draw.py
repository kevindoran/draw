# Our only import. 
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    def __init__(self, input_len, hidden_len):
        self.Wz = np.random.randn(hidden_len, input_len + hidden_len)
        self.Wr = np.random.randn(hidden_len, input_len + hidden_len)
        self.Wp = np.random.randn(hidden_len, input_len + hidden_len)
        self.bz = np.random.randn(hidden_len)
        self.br = np.random.randn(hidden_len)
        self.bp = np.random.randn(hidden_len)

    def forward(self, x, h):
        z = sigmoid(self.Wz @ np.concatenate([x, h]) + self.bz)
        r = sigmoid(self.Wr @ np.concatenate([x, h]) + self.br)
        p = np.tanh(self.Wp @ np.concatenate([x, r * h]) + self.bp)
        h_next = (1 - z) * p + z * h
        return h_next


class Draw:

    def __init__(self, img_shape, encode_hidden_len, latent_len, 
            decode_hidden_len):
        """Construct a Draw network.

        Args:
            img_shape (np.narray): shape of the input and generated images.
        """
        self.img_len = np.prod(img_shape)
        enc_input_len = self.img_len * 2
        self.enc_rnn = Gru(enc_input_len, encode_hidden_len)
        self.dec_rnn = Gru(latent_len, encode_hidden_len)
        # Optional: move the Gru hidden state vector into Gru class.
        #   * pros: encapsulate details relevant to GRU.
        #   * cons: slightly obscures how the network works.
        self.enc_h = np.zeros(encode_hidden_len)
        self.dec_h = np.zeros(decode_hidden_len)
        # Python 3 supports unicode variable names!
        self.W_μ = np.random.randn(latent_len, decode_hidden_len)
        self.b_μ = np.random.randn(latent_len)
        self.W_σ = np.random.randn(latent_len, decode_hidden_len)
        self.b_σ = np.random.randn(latent_len)
        self.W_write = np.random.randn(self.img_len, decode_hidden_len)
        self.b_write = np.random.randn(self.img_len)

    def _sample(self, e_override=None):
        e = np.random.standard_normal() if e_override is None else e_override
        μ = self.W_μ @ self.enc_h + self.b_μ
        log_σ = self.W_σ @ self.enc_h + self.b_σ
        σ = np.exp(log_σ)
        return (μ + σ*e)

    def _write(self, img):
        return img + self.W_write @ self.dec_h + self.b_write

    def forward(self, img_in, out_img=None):
        """Forward run of the draw network."""
        if out_img is None:
            out_img = np.zeros(self.img_len)
        # If using a prefix batch dimension, then use xx_hat =
        # np.concatenate(1, img_in, out_img)
        xx_hat = np.concatenate([img_in, out_img])
        self.h_enc = self.enc_rnn.forward(xx_hat, self.enc_h)
        z = self._sample()
        self.dec_h = self.dec_rnn.forward(z, self.dec_h)
        img_out = self._write(out_img)
        return img_out


