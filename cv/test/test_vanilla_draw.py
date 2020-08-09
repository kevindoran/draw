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


