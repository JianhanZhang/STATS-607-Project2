'''Implement unit tests for rejection rate computation.'''

import numpy as np
from src.metrics import compute_rejection_rate

def test_compute_rejection_rate_all_reject():
    '''Test rejection rate calculation when all tests reject.'''
    p_values = np.array([0.01, 0.012, 0.046, 0.049, 0.035, 0.04, 0.02, 0.021, 0.015, 0.01])   
    assert np.isclose(compute_rejection_rate(p_values, alpha=0.05), 1)

def test_compute_rejection_rate_no_reject():
    '''Test rejection rate calculation when no test rejects.'''
    p_values = np.array([0.01, 0.012, 0.046, 0.049, 0.035, 0.04, 0.02, 0.021, 0.015, 0.01])   
    assert np.isclose(compute_rejection_rate(p_values, alpha=0.01), 0)

def test_compute_rejection_rate_mixed_reject():
    '''Test rejection rate calculation when some tests reject and some tests fail to reject.'''
    p_values = np.array([0.01, 0.012, 0.046, 0.049, 0.035, 0.04, 0.02, 0.021, 0.015, 0.01])   
    assert np.isclose(compute_rejection_rate(p_values, alpha=0.02), 0.4)