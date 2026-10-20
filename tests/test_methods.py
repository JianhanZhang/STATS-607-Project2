'''Implement unit tests for the three tetrad tests.'''

import numpy as np
from scipy.stats import chi2
from src.dgps import generate_data
from src.methods import ClassicalTetradTest, GeneralizedTetradTestGMM, GeneralizedTetradTestPSMD

# ---------- TEST THE TEST STATISTIC ----------
def test_generalized_tetrad_gmm_statistic():
    '''Test that generalized tetrad test with GMM gives a finite and nonnegative test statistic.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0)
    test = GeneralizedTetradTestGMM(X, Y, Z, W)
    statistic = test.compute_test_statistic()

    assert np.isfinite(statistic)
    assert statistic >= 0

def test_generalized_tetrad_psmd_statistic():
    '''Test that generalized tetrad test with PSMD gives a finite and nonnegative test statistic.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0)
    test = GeneralizedTetradTestPSMD(X, Y, Z, W)
    statistic = test.compute_test_statistic()

    assert np.isfinite(statistic)
    assert statistic >= 0

# ---------- TEST THE P-VALUE ----------
def test_classical_tetrad_test_p_value_validity():
    '''Test that the classical tetrad test gives valid p-values.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = ClassicalTetradTest(X, Y, Z, W)
    p_value = test.compute_p_value()

    assert p_value >= 0 and p_value <= 1

def test_classical_tetrad_test_p_value_correctness():
    '''Test that the classical tetrad test gives correct p-values, assuming the test statistic is calculated correctly.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = ClassicalTetradTest(X, Y, Z, W)
    test_statistic = test.compute_test_statistic()
    p_value = test.compute_p_value()

    assert np.isclose(p_value, chi2.sf(test_statistic, df=2))

def test_generalized_tetrad_test_gmm_p_value_validity():
    '''Test that the generalized tetrad test with GMM gives valid p-values.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = GeneralizedTetradTestGMM(X, Y, Z, W)
    p_value = test.compute_p_value()

    assert p_value >= 0 and p_value <= 1

def test_generalized_tetrad_test_gmm_p_value_correctness():
    '''Test that the generalized tetrad test with GMM gives correct p-values, assuming the test statistic is calculated correctly.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = GeneralizedTetradTestGMM(X, Y, Z, W)
    test_statistic = test.compute_test_statistic()
    p_value = test.compute_p_value()

    assert np.isclose(p_value, chi2.sf(test_statistic, df=1))

def test_generalized_tetrad_test_psmd_p_value_validity():
    '''Test that the generalized tetrad test with PSMD gives valid p-values.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = GeneralizedTetradTestPSMD(X, Y, Z, W)
    p_value = test.compute_p_value()

    assert p_value >= 0 and p_value <= 1

def test_generalized_tetrad_test_psmd_p_value_correctness():
    '''Test that the generalized tetrad test with PSMD gives correct p-values, assuming the test statistic is calculated correctly.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = GeneralizedTetradTestPSMD(X, Y, Z, W)
    test_statistic = test.compute_test_statistic()
    p_value = test.compute_p_value()

    assert np.isclose(p_value, chi2.sf(test_statistic, df=1))

# ---------- TEST THE REJECTION DECISION ----------
def test_classical_tetrad_test_rejection_decision():
    '''Test that the classical tetrad test gives correct rejection decisions, assuming the p-value is calculated correctly.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = ClassicalTetradTest(X, Y, Z, W)
    p_value = test.compute_p_value()

    assert test.reject(alpha=0.10) == (p_value < 0.10)
    assert test.reject(alpha=0.05) == (p_value < 0.05)
    assert test.reject(alpha=0.01) == (p_value < 0.01)

def test_generalized_tetrad_test_gmm_rejection_decision():
    '''Test that the generalized tetrad test with GMM gives correct rejection decisions, assuming the p-value is calculated correctly.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = GeneralizedTetradTestGMM(X, Y, Z, W)
    p_value = test.compute_p_value()

    assert test.reject(alpha=0.10) == (p_value < 0.10)
    assert test.reject(alpha=0.05) == (p_value < 0.05)
    assert test.reject(alpha=0.01) == (p_value < 0.01)

def test_generalized_tetrad_test_psmd_rejection_decision():
    '''Test that the generalized tetrad test with PSMD gives correct rejection decisions, assuming the p-value is calculated correctly.'''
    X, Y, Z, W, U = generate_data(num_samples=100, alpha_2=0, beta_2=0, delta=0.2)
    test = GeneralizedTetradTestPSMD(X, Y, Z, W)
    p_value = test.compute_p_value()

    assert test.reject(alpha=0.10) == (p_value < 0.10)
    assert test.reject(alpha=0.05) == (p_value < 0.05)
    assert test.reject(alpha=0.01) == (p_value < 0.01)