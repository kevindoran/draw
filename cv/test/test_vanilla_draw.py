import draw.vanilla_draw
import numpy as np
import pytest


def test_gru_init():
    # Setup
    input_len = 4
    hidden_len = 7
    # Test
    # No errors should be thrown:
    gru = draw.vanilla_draw.Gru(input_len, hidden_len)


def test_gru_forward():
    # Setup
    input_len = 4
    hidden_len = 7
    gru = draw.vanilla_draw.Gru(input_len, hidden_len)
    test_input = np.random.randn(input_len)
    initial_state = np.random.randn(hidden_len)
    # Test 
    # No errors should be thrown:
    output = gru.forward(test_input, initial_state)


def test_draw_init():
    # Setup
    img_shape = (28, 28)
    encode_hidden_len = decode_hidden_len = 256
    latent_len = 15
    # Test
    # No errors should be thrown:
    net = draw.vanilla_draw.Draw(
        img_shape, encode_hidden_len, latent_len, decode_hidden_len)


def test_draw_forward():
    # Setup
    img_shape = (28, 28)
    encode_hidden_len = decode_hidden_len = 256
    latent_len = 15
    net = draw.vanilla_draw.Draw(
        img_shape, encode_hidden_len, latent_len, decode_hidden_len)
    # Note: maybe add some functions that conveniently initialize these 
    # vectors (add them to Draw)?
    test_img = np.random.randn(28 * 28) 
    prev_img = np.random.randn(28 * 28) 
    h_enc = np.random.randn(encode_hidden_len)
    h_dec = np.random.randn(decode_hidden_len)
    # Test
    # No errors should be thrown:
    updated_img = net.forward(test_img, prev_img, h_enc, h_dec)



