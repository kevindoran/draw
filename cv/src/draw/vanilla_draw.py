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

    def __init__(img_shape, encode_hidden_len, latent_len, decode_hidden_len):
        """Construct a Draw network.

        Args:
            img_shape (np.narray): shape of the input and generated images.
        """
        input_len = np.prod(img_shape)
        self.enc_rnn = Gru(input_len, encode_hidden_len)
        self.dec_rnn = Gru(latent_len, encode_hidden_len)
        # Optional: move the Gru hidden state vector into Gru class.
        #   * pros: encapsulate details relevant to GRU.
        #   * cons: slightly obscures how the network works.
        self.enc_h = np.random.randn(encode_hidden_len)
        self.dec_h = np.random.randn(decode_hidden_len)

    def _sample():
        # TODO
        # raise NotImplementedError()
        return np.random.randn(latent_len)

    def _write(img):
        raise NotImplementedError()

    def forward(img_in, out_img, h_enc, h_dec):
        """Forward run of the draw network."""
        # Switching to variable names that match the DRAW network diagram.
        x = img_in
        x_hat = img_out

        xx_hat = np.concatenate(x, h_hat)
        self.h_enc = self.enc_rnn.forward(xx_hat, self.enc_h)
        z = self.sample()
        self.h_dec = self.dec_rnn.forward(z, self.h_dec)
        img_out = self.write(out_img)
        return img_out


