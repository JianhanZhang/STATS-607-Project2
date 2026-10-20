'''Implement unit tests for the data-generating mechanism.'''

import numpy as np
from src.dgps import generate_data

def test_generate_data_type():
    '''Test whether the generated data has expected data type.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray) 
    assert isinstance(Z, np.ndarray)
    assert isinstance(W, np.ndarray)
    assert isinstance(U, np.ndarray)

def test_generate_data_size():
    '''Test whether the generated data has expected size.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)

    assert X.shape == (100,)
    assert Y.shape == (100,)
    assert Z.shape == (100,)
    assert W.shape == (100,)
    assert U.shape == (100,)