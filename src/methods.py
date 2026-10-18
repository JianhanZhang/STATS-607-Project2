from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import chi2

class TetradTest(ABC):
    '''
    Base class for the tetrad tests being compared in the simulation study.
    '''

    def __init__(self, X, Y, Z, W):
        '''
        Initialize the TetradTest with the input variables.

        Parameters:
        X, Y, Z, W (np.ndarray): Input variables for the tetrad test.
        '''
        self.X = X
        self.Y = Y
        self.Z = Z
        self.W = W

        self._sanity_check()

        self.N = len(X)

    def _sanity_check(self):
        '''
        Perform sanity checks on the input variables to ensure they have valid shapes.

        Raises:
        ValueError: If any of the input variables are not one-dimensional or have different lengths.
        '''
        if not (self.X.ndim == 1):
            raise ValueError("X must be a one-dimensional array.")
        if not (self.Y.ndim == 1):
            raise ValueError("Y must be a one-dimensional array.")
        if not (self.Z.ndim == 1):
            raise ValueError("Z must be a one-dimensional array.")
        if not (self.W.ndim == 1):
            raise ValueError("W must be a one-dimensional array.")
        if not (len(self.X) == len(self.Y) == len(self.Z) == len(self.W)):
            raise ValueError("All input variables must have the same length.")

    @abstractmethod
    def compute_test_statistic(self):
        '''
        Abstract method to compute the test statistic for the tetrad test.

        Returns:
        float: The computed test statistic.
        '''
        pass

    @abstractmethod
    def compute_p_value(self):
        '''
        Abstract method to compute the p-value for the tetrad test based on the test statistic.

        Returns:
        float: The computed p-value.
        '''
        pass

    def reject(self, alpha=0.05):
        '''
        Determine whether to reject the null hypothesis based on the computed p-value.

        Parameters:
        alpha (float): Significance level for the test.

        Returns:
        bool: True if the null hypothesis is rejected, False otherwise.
        '''
        p_value = self.compute_p_value()
        return p_value < alpha



class ClassicalTetradTest(TetradTest):
    '''
    Implement the classical tetrad test.
    '''

    def _compute_covariance_matrix(self):
        '''
        Compute the sample covariance matrix for the input variables.

        Returns:
        np.ndarray: The computed sample covariance matrix.
        '''
        return np.cov(np.column_stack([self.X, self.Y, self.Z, self.W]), rowvar=False, ddof=0)

    def _compute_tetrad_differences(self, cov_matrix):
        '''
        Compute the vector of tetrad differences based on the covariance matrix.

        Parameters:
        cov_matrix (np.ndarray): Sample covariance matrix of the input variables.

        Returns:
        np.ndarray: The computed tetrad differences.
        '''
        cov_xy = cov_matrix[0, 1]
        cov_xz = cov_matrix[0, 2]
        cov_xw = cov_matrix[0, 3]
        cov_yz = cov_matrix[1, 2]
        cov_yw = cov_matrix[1, 3]
        cov_zw = cov_matrix[2, 3]

        return np.array([
            cov_xy * cov_zw - cov_xz * cov_yw, 
            cov_xy * cov_zw - cov_xw * cov_yz
        ])

    def _compute_covariance_of_sample_covariances(self, cov_matrix):
        '''
        Compute the covariance matrix of the sample covariances.

        Parameters:
        cov_matrix (np.ndarray): The sample covariance matrix.

        Returns:
        np.ndarray: The covariance matrix of the sample covariances.
        '''
        # Center the variables
        combined_data = np.column_stack([self.X, self.Y, self.Z, self.W])
        centered_data = combined_data - np.mean(combined_data, axis=0)

        # compute the covariance of sample covariances
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        centered_covariances = np.zeros((self.N, 6))
        for idx, pair in enumerate(pairs):
            i, j = pair
            centered_covariances[:, idx] = centered_data[:, i] * centered_data[:, j] - cov_matrix[i, j]
        covariance_of_sample_covariances = centered_covariances.T @ centered_covariances / self.N

        return covariance_of_sample_covariances

    def _compute_covariance_of_tetrad_differences(self, cov_matrix, cov_of_sample_covariances):
        '''
        Compute the covariance matrix of the tetrad differences with respect to the sample covariances.

        Parameters:
        cov_matrix (np.ndarray): Sample covariance matrix of the input variables.
        cov_of_sample_covariances (np.ndarray): Covariance matrix of the sample covariances.

        Returns:
        np.ndarray: The covariance matrix of the tetrad differences.
        '''
        cov_xy = cov_matrix[0, 1]
        cov_xz = cov_matrix[0, 2]
        cov_xw = cov_matrix[0, 3]
        cov_yz = cov_matrix[1, 2]
        cov_yw = cov_matrix[1, 3]
        cov_zw = cov_matrix[2, 3]

        jacobian = np.zeros((2, 6))
        jacobian[0, 0] = cov_zw
        jacobian[0, 1] = -cov_yw
        jacobian[0, 2] = 0
        jacobian[0, 3] = 0
        jacobian[0, 4] = -cov_xz
        jacobian[0, 5] = cov_xy

        jacobian[1, 0] = cov_zw
        jacobian[1, 1] = 0
        jacobian[1, 2] = -cov_yz
        jacobian[1, 3] = -cov_xw
        jacobian[1, 4] = 0
        jacobian[1, 5] = cov_xy

        return jacobian @ cov_of_sample_covariances @ jacobian.T

    def compute_test_statistic(self):
        '''
        Compute the test statistic for the classical tetrad test.

        Returns:
        float: The computed test statistic.
        '''
        # Compute the sample covariance matrix
        cov_matrix = self._compute_covariance_matrix()

        # Compute the tetrad differences
        tetrad_differences = self._compute_tetrad_differences(cov_matrix)

        # Compute the covariance of the sample covariances
        covariance_of_sample_covariances = self._compute_covariance_of_sample_covariances(cov_matrix)

        # Compute the covariance of the tetrad differences
        tetrad_diffs_covariance = self._compute_covariance_of_tetrad_differences(cov_matrix, covariance_of_sample_covariances)

        # Compute the test statistic
        test_statistic = self.N * (tetrad_differences.T @ np.linalg.solve(tetrad_diffs_covariance, tetrad_differences))

        return test_statistic

    def compute_p_value(self):
        '''
        Compute the p-value for the classical tetrad test based on the test statistic.

        Returns:
        float: The computed p-value.
        '''

        # Compute the test statistic
        test_statistic = self.compute_test_statistic()
        
        # Compare the test statistic to a chi-squared distribution with 2 degrees of freedom
        p_value = chi2.sf(test_statistic, df=2)

        return p_value