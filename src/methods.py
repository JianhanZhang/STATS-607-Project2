from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.optimize import minimize

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

    @abstractmethod
    def reject(self, alpha=0.05):
        '''
        Abstract method to determine whether to reject the null hypothesis based on the computed p-value.

        Parameters:
        alpha (float): Significance level for the test.

        Returns:
        bool: True if the null hypothesis is rejected, False otherwise.
        '''
        pass



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

class GeneralizedTetradTestGMM(TetradTest):
    '''
    Implement the generalized tetrad test, with the bridge function estimated by GMM.
    '''

    def _estimate_bridge_function(self, bridge_function_type):
        '''
        Estimate the bridge function using GMM. The bridge function is assumed to be linear 
        i.e. B = theta_0 + theta_1 * A

        Parameters:
        bridge_function_type (str): The type of bridge function to be estimated. If "h", then the Y-W bridge function is estimated; 
            if "g", then the Y-Z bridge function is estimated.

        Returns:
        callable: The estimated bridge function.
        '''
        # Sanity check
        if bridge_function_type not in ["h", "g"]:
            raise ValueError("bridge_function_type must be either 'h' or 'g'.")

        if bridge_function_type == "h":
            A = self.W
            instrument = self.Z
        else:
            A = self.Z
            instrument = self.W

        # Moment function used for GMM
        def moment_function(theta):
            theta_0, theta_1 = theta
            B_hat = theta_0 + theta_1 * A
            moments = np.column_stack([
                (self.Y - B_hat),
                (self.Y - B_hat) * instrument
            ])
            return moments

        # Function to compute the GMM objective
        def gmm_objective(theta, moment_function, weight_matrix):
            moments = moment_function(theta)
            mean_moments = np.mean(moments, axis=0)
            return mean_moments.T @ weight_matrix @ mean_moments

        # Estimate the parameter of the bridge function
        result = minimize(
                    gmm_objective,
                    x0=np.zeros(2),
                    args=(moment_function, np.eye(2)),
                    method="BFGS"
                 )
        theta_hat = result.x

        # Function that represents the bridge function based on the estimated parameters
        def estimated_bridge_function(C):
            return theta_hat[0] + theta_hat[1] * C

        return estimated_bridge_function

    def _compute_delta_matrix(self):
        '''
        Compute the delta matrix as defined in equation (11) in the generalized tetrad paper.

        Returns:
        np.ndarray: The computed delta matrix.
        '''
        abs_diff_matrix = np.abs(np.subtract.outer(self.X, self.X))
        term_1 = np.add.outer(np.sum(abs_diff_matrix, axis=0), np.sum(abs_diff_matrix, axis=1)) / self.N
        term_2 = np.sum(abs_diff_matrix) / (self.N)**2

        return abs_diff_matrix - term_1 + term_2

    def _compute_mgtn(self, delta_matrix, bridge_function, bridge_function_type):
        '''
        Compute the mgtn vector as defined in equation (12) in the generalized tetrad paper.

        Parameters:
        delta_matrix (np.ndarray): The computed delta matrix.
        bridge_function (callable): The estimated bridge function.
        bridge_function_type (str): The type of bridge function to be used. If "h", then the Y-W bridge function is used; 
            if "g", then the Y-Z bridge function is used.

        Returns:
        np.ndarray: The computed mgtn vector.
        '''
        # Sanity check
        if bridge_function_type not in ["h", "g"]:
            raise ValueError("bridge_function must be either 'h' or 'g'.")

        if bridge_function_type == "h":
            self.A = self.W
        else:
            self.A = self.Z

        # Compute the residuals R(h) = Y - h(A) or R(g) = Y - g(A), and center them
        residuals = self.Y - bridge_function(self.A)
        centered_residuals = residuals - np.mean(residuals)

        # Compute the quadratic form inside the square root of MGT_n
        quadratic_form = (centered_residuals / self.N).T @ delta_matrix @ (centered_residuals / self.N)

        return np.sqrt(-quadratic_form)

    def _compute_phi_s(self, s, bridge_function, bridge_function_type):
        """
        Compute phi_hat_s(O_j; bridge) as defined in Section 4.2 of the generalized tetrad paper.

        Parameters:
        s (float): The value of s for which to compute phi_hat_s.
        bridge_function (callable): The estimated bridge function.
        bridge_function_type (str): The type of bridge function to be used. If "h", then the Y-W bridge function is used; if "g", then the Y-Z bridge function is used.

        Returns:
        np.ndarray: The computed phi_hat_s(O_j; bridge) for all observations, with shape (N,).
        """
        if bridge_function_type == "h":
            proxy = self.W
            instrument = self.Z
        elif bridge_function_type == "g":
            proxy = self.Z
            instrument = self.W
        else:
            raise ValueError("bridge_function_type must be 'h' or 'g'.")

        # Calculate GMM moment functions
        D = np.column_stack([np.ones(self.N), proxy])
        B = np.column_stack([np.ones(self.N), instrument])

        residuals = self.Y - bridge_function(proxy)
        J = residuals[:, None] * B

        # Estimate Sigma_1
        M = -(B.T @ D) / self.N

        # Sigma_1 = M^{-1} because the GMM is just identified
        Sigma_1 = np.linalg.solve(M, np.eye(2))

        # Calculate the exponentiated values
        exp_values = np.exp(1j * s * self.X)

        C = (
            D.T @ exp_values / self.N
            - D.mean(axis=0) * exp_values.mean()
        )

        # Step 4: Compute phi_hat_s for all observations
        phi_s = C @ Sigma_1 @ J.T

        return phi_s

    def _compute_sn(self, bridge_function, bridge_function_type, s_grid=None):
        '''
        Compute S_n as defined in Equation (12) in the generalized tetrad paper.

        Parameters:
        bridge_function (callable): The estimated bridge function.
        bridge_function_type (str): The type of bridge function to be used. If "h", then the Y-W bridge function is used; if "g", then the Y-Z bridge function is used.
        s_grid (np.ndarray, optional): A grid of s values for numerical integration. If None, a default grid will be used.

        Returns:
        float: The computed S_n value.
        '''
        if bridge_function_type == "h":
            proxy = self.W
        elif bridge_function_type == "g":
            proxy = self.Z
        else:
            raise ValueError("Invalid bridge function type.")

        residuals = self.Y - bridge_function(proxy)

        if s_grid is None:
            s_grid = np.linspace(0.001, 300, 10000)

        integrand = np.zeros(len(s_grid))

        for idx, s in enumerate(s_grid):
            exp_values = np.exp(1j * s * self.X)

            # Compute phi_hat_s(O_j; bridge) for all observations
            phi_s = self._compute_phi_s(
                s, bridge_function, bridge_function_type
            )

            # Compute the expression inside absolute value in Equation (12)
            terms = (
                residuals * (exp_values - exp_values.mean())
                + phi_s
            )

            integrand[idx] = np.sum(np.abs(terms)**2) / s**2

        # Integrate over positive s and multiply by 2
        sn = (2 / (self.N * np.pi)) * np.trapezoid(
            integrand, s_grid
        )

        return sn

    def compute_test_statistic(self):
        '''
        Compute the test statistic for the generalized tetrad test.

        Returns:
        float: The computed test statistic.
        '''
        # Estimate the bridge functions
        h_hat = self._estimate_bridge_function(bridge_function_type="h")
        g_hat = self._estimate_bridge_function(bridge_function_type="g")

        # Compute the delta matrix
        delta_matrix = self._compute_delta_matrix()

       # Compute MGT_n for both bridge functions
        mgtn_h = self._compute_mgtn(delta_matrix, h_hat, bridge_function_type="h")
        mgtn_g = self._compute_mgtn(delta_matrix, g_hat, bridge_function_type="g")

        # Compute S_n for both bridge functions
        sn_h = self._compute_sn(h_hat, bridge_function_type="h")
        sn_g = self._compute_sn(g_hat, bridge_function_type="g")

        # Compute the test statistic as the maximum of the two S_n values
        test_statistic = self.N * (mgtn_h**2 + mgtn_g**2) / (sn_h + sn_g)

        return test_statistic

    def compute_p_value(self):
        '''
        Compute the p-value for the generalized tetrad test based on the test statistic.

        Returns:
        float: The computed p-value.
        '''
        # Compute the test statistic
        test_statistic = self.compute_test_statistic()

        # Compare the test statistic to a chi-squared distribution with 1 degrees of freedom
        p_value = chi2.sf(test_statistic, df=1)

        return p_value

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
