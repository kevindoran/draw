# Our only import. 
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Model parameters
class Gru:
    """Encapsulates the state and behaviour of a GRU (gated recurrent unit).

    Steps carried out within a GRU network:

    (1) z = σ(Wzx x_t + Wzh h_{t-1} + bz)
    (2) r = σ(Wrx x_t + Wrh h_{t-1} + br)
    (3) p = tanh(Wpx x_t + Wph r.h_{t-1})
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

    From steps (1)-(4), we see that there are a total of 6 trainable matricies
    and 2 trainable bias vectors:

        Matricies: Wzx, Wzh, Wrx, Wrh, Wpx, Wph
        Bias vectors: bz, br
    """
    def __init__(self, input_len, hidden_len):
        self.Wzx = np.random.randn(hidden_len, input_len) # *0.01?
        self.Wzh = np.random.randn(hidden_len, hidden_len)
        self.Wrx = np.random.randn(hidden_len, input_len)
        self.Wrh = np.random.randn(hidden_len, hidden_len)
        self.Wpx = np.random.randn(hidden_len, input_len)
        self.Wph = np.random.randn(hidden_len, hidden_len)

    def forward(self, x, h):
        z = sigmoid(self.Wzx.dot(x) + self.Wzh.dot(h))
        r = sigmoid(self.Wrx.dot(x) + self.Wrh.dot(h))
        h_reset = r.dot(self.Wph)
        p = np.tanh(self.Wpx.dot(x) + self.Wph.dot(h_reset))
        h_next = (1 - z).dot(p) + z.dot(h)
        return h_next


class Draw:

    def __init__(self, img_shape, encode_hidden_len, latent_len, 
            decode_hidden_len):
        """Construct a Draw network.

        Args:
            img_shape (np.narray): shape of the input and generated images.
        """
        input_len = np.prod(img_shape)
        enc_input_len = input_len * 2
        self.enc_rnn = Gru(enc_input_len, encode_hidden_len)
        self.dec_rnn = Gru(latent_len, encode_hidden_len)
        # Optional: move the Gru hidden state vector into Gru class.
        #   * pros: encapsulate details relevant to GRU.
        #   * cons: slightly obscures how the network works.
        self.enc_h = np.random.randn(encode_hidden_len)
        self.dec_h = np.random.randn(decode_hidden_len)
        # Python 3 supports unicode variable names!
        self.W_μ = np.random.randn(latent_len, decode_hidden_len)
        self.b_μ = np.random.randn(latent_len)
        self.W_σ = np.random.randn(latent_len, decode_hidden_len)
        self.b_σ = np.random.randn(latent_len)
        self.W_write = np.random.randn(latent_len, input_len)
        self.b_write = np.random.randn(input_len)

    def _sample(self):
        μ = self.W_μ.dot(self.enc_h) + self.b_μ
        log_σ = self.W_σ.dot(self.enc_h) + self.b_σ
        σ = np.exp(log_σ)
        e = np.random.standard_normal()
        return (μ + σ*e)

    def _write(self, img):
        return img + self.W_write.dot(self.dec_h) + self.b_write

    def forward(self, img_in, out_img, h_enc, h_dec):
        """Forward run of the draw network."""
        # If using a prefix batch dimension, then use
        # xx_hat = np.concatenate(1, img_in, out_img)
        xx_hat = np.concatenate([img_in, out_img])
        self.h_enc = self.enc_rnn.forward(xx_hat, self.enc_h)
        z = self._sample()
        self.dec_h = self.dec_rnn.forward(z, self.dec_h)
        img_out = self._write(out_img)
        return img_out


