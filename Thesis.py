import numpy as np
import pandas as pd
import math
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import time
import sys
from scipy.optimize import minimize
from datetime import timedelta
from numpy.linalg import norm

# Epanechnikov kernel function
def Epanechnikov(u):
    """
    Epanechnikov kernel function for a single value.
    Returns:
    Epanechnikov kernel values
    """
        
    K = np.where(np.abs(u) <= 1, 3/4 * (1 - u**2), 0)
    
    return K

# Kernel weighting function with bandwidth h
def kernel_weights(tau_t, tau, h):
    """
    Compute Epanechnikov kernel weights based on the time difference.
    Returns:
    Weights for each time point in tau_t.
    """
    return Epanechnikov((tau_t - tau) / h) / h

# Create lagged data matrix Xt-1
def create_lagged_matrix(X):
    """
    Creates the lagged data matrix Xt-1 from the input data X.
    
    Parameters:
    - X: (n × d) matrix of observations.
    
    Returns:
    - Xt_1: (n-1 × d) lagged matrix of predictors.
    """
    n, d = X.shape
    lagged_matrix = np.zeros((n-1, d))
    for t in range(1, n):
        lagged_matrix[t-1] = X[t-1]
    return lagged_matrix

# Compute effective sample size
def effective_sample_size(tau_t_values, tau, bandwidth):
    """
    Compute the effective sample size n_e for a given set of time points tau_t_values,
    reference point tau, and bandwidth.
    
    n_e = Σ Kh(tau_t - tau) / max_t{Kh(tau_t - tau)}
    """
    # Compute kernel weights Kh(tau_t - tau) for each time point
    weights_array = np.array([kernel_weights(tau_t, tau, bandwidth) for tau_t in tau_t_values])
    
    # Calculate the effective sample size
    sum_of_weights = np.sum(weights_array)
    max_weight = np.max(weights_array)
    
    # Effective sample size n_e
    n_e = sum_of_weights / max_weight
    
    return n_e

def BIC_i(λ1, alpha_hat, beta_hat, tau, tau_t_values, X, Xt_1, h):
    """
    Compute the Bayesian Information Criterion (BIC) for the given estimates of alpha and beta.
    
    Parameters:
    - λ1: Tuning parameter for LASSO.
    - alpha_hat: Estimated alpha coefficients.
    - beta_hat: Estimated beta coefficients.
    - tau: Target time point.
    - tau_t_values: Time points.
    - X: Response matrix (n × d).
    - Xt_1: Lagged predictors (n-1 × d).
    - h: Bandwidth for kernel smoothing.
    
    Returns:
    - BIC value.
    """
    n, d = X.shape
    
    # Compute the effective sample size n_e
    n_e = effective_sample_size(tau_t_values, tau, h)
    
    # Compute the kernel weights for all time points
    weights = kernel_weights(tau_t_values, tau, h)
    
    # Initialize the likelihood term
    L_i = 0
    
    # Compute the likelihood term (L_i)
    for t in range(n):
        y_t = X[t, :]
        predicted = X[t, :] @ np.hstack([alpha_hat, beta_hat * (tau_t_values[t] - tau)])
        residual = y_t - predicted
        
        # Update the likelihood with the weighted residual
        L_i += np.sum(residual ** 2) * weights[t]

    # Normalize by the total number of observations
    L_i /= n
    
    # Number of non-zero coefficients
    non_zero_alpha = np.count_nonzero(alpha_hat)
    non_zero_beta = np.count_nonzero(beta_hat)
    
    # Compute the BIC
    BIC_value = np.log(L_i / np.sum(weights[1:])) + ((np.log(n_e) / n_e)) * (non_zero_alpha + non_zero_beta)
    
    return BIC_value

def local_linear_lasso(X, Xt_1, tau, tau_t, h, lambda_1):
    """
    Perform local linear regression with LASSO regularization.
    
    Parameters:
    - X: (n × d) matrix of observations (response data).
    - Xt_1: (n-1 × d) matrix of lagged predictors.
    - tau: Target time for local regression (scalar in [0, 1]).
    - tau_t: Normalized time points (n × 1 array).
    - h: Bandwidth parameter for kernel smoothing.
    - λ: Regularization parameter for LASSO.
    
    Returns:
    - Estimated alpha and beta parameters for each variable.
    """
    n, d = X.shape
    alpha_results = []  # List to store alpha estimates for each time point
    beta_results = []  # List to store beta estimates for each time point

    # For each variable i, we will estimate alpha and beta
    for i in range(d):
        # Step 1: Create the response vector and the design matrix
        y = X[1:, i]  # Response vector for variable i (exclude the first row)
        design_matrix = np.hstack([Xt_1, (tau_t[1:] - tau).reshape(-1, 1) * Xt_1])
        
        # Step 2: Calculate kernel weights
        weights = kernel_weights(tau_t[1:], tau, h)  # Kernel weights for each time point (exclude first)
        
        # Step 3: Perform weighted least squares using LASSO
        weighted_design_matrix = design_matrix * np.sqrt(weights).reshape(-1, 1)
        weighted_y = y * np.sqrt(weights)
        
        # Step 4: Custom LASSO with separate penalties for alpha and beta
        scaling_factor = np.hstack([np.ones(d), h * np.ones(d)])  # alpha penalized by λ, beta by hλ
        scaled_design_matrix = weighted_design_matrix / scaling_factor
        
        # Step 5: Fit LASSO (L1 regularized linear regression)
        lasso = Lasso(alpha=lambda_1, fit_intercept=False)
        lasso.fit(scaled_design_matrix, weighted_y)
        
        # Step 6: Extract the coefficients (alpha and beta)
        alpha = lasso.coef_[:d]  # First d coefficients correspond to alpha
        beta = lasso.coef_[d:] / h  # Next d coefficients correspond to beta (scaled back by h)
        
        # Store the result for the current variable i
        alpha_results.append(alpha)
        beta_results.append(beta)
    
    return np.array(alpha_results), np.array(beta_results)

# SCAD penalty derivative
def scad_penalty_derivative(lambda_, z, a0):
    """
    Computes the derivative of the SCAD penalty function.
    """
    if z <= lambda_:
        return lambda_
    elif z <= a0 * lambda_:
        return (a0 * lambda_ - z) / (a0 - 1)
    else:
        return 0
    
# Compute D hat
def compute_D_hat(alpha_hat):
    """
    Computes the D_hat^2 values for each predictor j based on preliminary 
    LASSO estimates alpha_hat.
    
    Parameters:
    - alpha_hat: A numpy array of shape (n, d, d) containing the estimates of alpha.

    Returns:
    - D_hat: A numpy array of shape (d, d) containing the computed D_hat^2 values.
    """
    n, d, _ = alpha_hat.shape
    D_hat = np.zeros((d, d))  # Initialize D_hat with the shape (d, d)

    # Compute the mean of alpha over all time points for each predictor j
    alpha_mean = np.mean(alpha_hat, axis=0)  # Shape (d, d)

    # Compute the sum of squared deviations for each predictor j
    for i in range(d):  # Iterate over predictors
        for j in range(d):  # Iterate over responses
            D_hat[i, j] = np.sum((alpha_hat[:, i, j] - alpha_mean[i, j]) ** 2)

    return D_hat
     
# Local linear objective function
def local_linear_objective(alpha_t, beta_t, Xt_1, xt_i, tau_t, tau, h):
    """
    Compute the local linear objective function L_i(alpha, beta | tau).

    Parameters:
    - alpha_t: Coefficients alpha at time t.
    - beta_t: Coefficients beta at time t.
    - Xt_1: Lagged predictor matrix at time t-1.
    - xt_i: Predictor data at time t for variable i.
    - tau_t: Current time point.
    - tau: Reference time point.
    - bandwidth: Bandwidth for kernel smoothing.

    Returns:
    - Local linear objective value.
    """
    prediction = alpha_t + beta_t * (tau_t - tau)  # Linear prediction
    loss = np.sum((xt_i - Xt_1 @ prediction) ** 2)  # Squared loss
    weight = kernel_weights(tau_t, tau, h)  # Get the kernel weight
    return (1 / len(Xt_1)) * loss * weight

def compute_penalized_objective(preliminary_a, preliminary_b, D_hat, X, Xt_1, tau_t, λ, a0, h):
    """
    Compute the global penalized objective function with weighted group LASSO.
    
    Parameters:
    - preliminary_a: Preliminary alpha estimates (shape: n, d, d).
    - preliminary_b: Preliminary beta estimates (shape: n, d, d).
    - D_hat: Pre-computed D_hat matrix.
    - X: Predictor data (shape: n, d).
    - Xt_1: Lagged predictor data (shape: n-1, d).
    - tau_t: Time points (shape: n, 1).
    - λ: Regularization parameter for SCAD.
    
    Returns:
    - The penalized objective value Q_i(A, B).
    """
    n, d = X.shape  # n = number of time points, d = number of variables
    total_loss = 0
    total_penalty_alpha = 0
    total_penalty_beta = 0
    
    # Initialize estimated estimates for alpha and beta (A and B)
    alpha_estimates = np.zeros_like(preliminary_a)
    beta_estimates = np.zeros_like(preliminary_b)

    # Iterate over each time point (excluding the first due to lagged matrix Xt_1)
    for t in range(n-1):
        alpha_t = preliminary_a[t]  # alpha estimates at time t (shape: d, d)
        beta_t = preliminary_b[t]  # beta estimates at time t (shape: d, d)
        
        Xt_1_t = Xt_1[t]  # Lagged predictor matrix Xt_1 at time t
        xt_i = X[t + 1]  # Predictor data at time t (to match Xt_1)
        
        # Local linear loss term
        total_loss += local_linear_objective(alpha_t, beta_t, Xt_1_t, xt_i, tau_t[t + 1], tau_t[t], h)
    
        # Refining the estimates of A (alpha) and B (beta) based on SCAD penalty
        for i in range(d):
            for j in range(d):
                # Refine alpha estimates
                norm_alpha_ij = np.linalg.norm(preliminary_a[:, i, j])
                penalty_alpha = scad_penalty_derivative(λ, norm_alpha_ij, a0)
                alpha_estimates[t, i, j] = np.sign(alpha_t[i, j]) * max(0, abs(alpha_t[i, j]) - penalty_alpha)

                # Refine beta estimates with D_hat weighting
                penalty_beta = scad_penalty_derivative(λ, D_hat[i, j], a0)
                beta_estimates[t, i, j] = np.sign(beta_t[i, j]) * max(0, abs(beta_t[i, j]) - penalty_beta)
        
    # Penalty terms for alpha and beta using SCAD and D_hat
    for i in range(d):  # Iterate over each variable i
        for j in range(d):  # Iterate over each variable j (alpha and beta are d × d matrices)
            # Penalty for alpha
            norm_alpha_ij = np.linalg.norm(preliminary_a[:, i, j])  # Norm over all time points for alpha_i,j
            penalty_alpha = scad_penalty_derivative(λ, norm_alpha_ij, a0)
            total_penalty_alpha += penalty_alpha * norm_alpha_ij
            
            # Penalty for beta with D_hat weighting
            norm_beta_ij = np.linalg.norm(preliminary_b[:, i, j])  # Norm over all time points for beta_i,j
            penalty_beta = scad_penalty_derivative(λ, D_hat[i, j], a0)
            total_penalty_beta += penalty_beta * norm_beta_ij

    # Final objective function value (sum of loss and penalties)
    Q_i = total_loss + total_penalty_alpha + total_penalty_beta
    
    # Return the penalized objective and the estimated alpha and beta results
    return Q_i, alpha_estimates, beta_estimates

# Print estimated alpha and beta matrices from step 3.2    
def print_estimated_matrices(alpha_estimates, beta_estimates, tau_t_values):
    """
    Prints the estimated alpha and beta transition matrices for each time point tau.
    
    Parameters:
    - alpha_estimates: estimated alpha estimates (shape: n, d, d).
    - beta_estimates: estimated beta estimates (shape: n, d, d).
    - tau_t_values: Time points (shape: n, 1).
    """
    n, d, _ = alpha_estimates.shape  # n is the number of time points, d is the number of variables
    
    # Iterate through all time points to print estimated transition matrices
    for t in range(n):
        print(f"\nTime point tau = {tau_t_values[t]:.4f}")
        print(f"estimated alpha (A) matrix at tau = {tau_t_values[t]:.4f}:\n{alpha_estimates[t]}")
        print(f"estimated beta (B) matrix at tau = {tau_t_values[t]:.4f}:\n{beta_estimates[t]}")

# Estimate residuals
def estimate_residuals_with_estimated_A(X, alpha_estimates, p=1):
    """
    Estimate residuals (errors) e_t for each time point using the estimated transition matrices.

    Parameters:
    - X: (n × d) matrix of observations (the original data).
    - estimated_alpha_results: (n × d × d) matrix of estimated time-varying transition matrices.
    - p: Number of lags in the model (for simplicity, we assume p = 1).

    Returns:
    - residuals: (n, d) matrix of estimated residuals e_t.
    """
    n, d = X.shape
    residuals = np.zeros((n, d))
    
    # Loop over each time point t, starting from time point p (because of lags)
    for t in range(p, n):
        # Initialize predicted_X_t as a zero vector for d variables
        predicted_X_t = np.zeros(d)
        
        # Compute the predicted value of X_t based on the estimated transition matrix at time t
        # Using the transition matrix for the current time point
        predicted_X_t = alpha_estimates[t] @ X[t-1]  # estimated_alpha_results[t] is A(tau_t), X[t-1] is the lagged value
        
        # Calculate the residual (error) as the difference between the actual and predicted X_t
        residuals[t] = X[t] - predicted_X_t
    
    return residuals

def estimate_sigma_tau(residuals, tau_t_values, bandwidth):
    """
    Estimate the time-varying covariance matrix Σ(tau) for each time point tau.

    Parameters:
    - residuals: (n × d) matrix of residuals e_t from the previous step.
    - tau_t_values: Array of normalized time points (n values).
    - bandwidth: Bandwidth parameter for the kernel smoothing.

    Returns:
    - sigma_tau: A list of (d × d) covariance matrices Σ(tau) for each time point tau.
    """
    n, d = residuals.shape
    sigma_tau = []

    # Loop over each time point tau
    for tau in tau_t_values:
        # Initialize the weighted covariance matrix
        sigma_tau = np.zeros((d, d))
        
        # Compute the kernel weights for each time point t
        weights = kernel_weights(tau_t_values, tau, bandwidth)  # Shape: (n,)
        
        # Compute the weighted covariance matrix at tau
        for i in range(d):
            for j in range(d):
                # Numerator: Sum of weighted residual products
                weighted_sum = np.sum(weights * residuals[:, i] * residuals[:, j])
                
                # Denominator: Sum of the weights
                sum_of_weights = np.sum(weights)
                
                # Calculate the covariance element σ_ij(tau)
                sigma_tau[i, j] = weighted_sum / sum_of_weights
        
        # Append the covariance matrix for time point tau
        sigma_tau.append(sigma_tau)
    
    return sigma_tau

def clime_estimation(Sigma_tau, lambda_3):
    """
    Perform CLIME estimation to get the precision matrix Omega(tau) for each time point tau.

    Parameters:
    - Sigma_tau: The covariance matrix Σ(tau) (d × d).
    - lambda_3: Tuning parameter for CLIME.

    Returns:
    - Omega_hat: The estimated precision matrix Ω(tau) (d × d).
    """
    d = Sigma_tau.shape[0]
    identity_matrix = np.eye(d)

    # Define the objective function (L1 norm of Omega)
    def objective_function(Omega):
        Omega = Omega.reshape((d, d))
        return norm(Omega, ord=1)  # L1 norm of the precision matrix

    # Define the constraint: ||Σ(tau) * Omega - I_d||_max <= λ3
    def constraint_function(Omega):
        Omega = Omega.reshape((d, d))
        return norm(Sigma_tau @ Omega - identity_matrix, ord=np.inf) - lambda_3

    # Initial guess (identity matrix for precision matrix)
    Omega_initial = np.eye(d).flatten()

    # Solve the optimization problem with constraints
    result = minimize(
        objective_function,
        Omega_initial,
        constraints={"type": "ineq", "fun": constraint_function},
        options={"disp": False},
        method="SLSQP"
    )

    # Reshape the result back into a matrix form (d × d)
    Omega_hat = result.x.reshape((d, d))
    return Omega_hat

# Estimate precision matrices
def estimate_precision_matrices(sigma_tau_list, lambda_3):
    """
    Estimate time-varying precision matrices Ω(tau) for each time point tau.

    Parameters:
    - sigma_tau_list: List of (d × d) covariance matrices Σ(tau) for each time point.
    - lambda_3: Tuning parameter for CLIME.

    Returns:
    - omega_tau_list: List of (d × d) precision matrices Ω(tau) for each time point.
    """
    omega_tau_list = []

    # Loop over each covariance matrix Σ(tau)
    for Sigma_tau in sigma_tau_list:
        # Estimate the precision matrix Ω(tau)
        Omega_hat = clime_estimation(Sigma_tau, lambda_3)
        omega_tau_list.append(Omega_hat)
    
    return omega_tau_list

# Symmetrize precision matrix
def symmetrize_precision_matrix(Omega_tau):
    """
    Symmetrize the estimated precision matrix Ω(tau).

    Parameters:
    - Omega_tau: The estimated precision matrix Ω(tau) (d × d).

    Returns:
    - Omega_sym: The symmetrized precision matrix (d × d).
    """
    d = Omega_tau.shape[0]
    Omega_sym = np.zeros((d, d))

    # Apply the symmetrization rule for each element
    for i in range(d):
        for j in range(d):
            if np.abs(Omega_tau[i, j]) <= np.abs(Omega_tau[j, i]):
                Omega_sym[i, j] = Omega_tau[i, j]
            else:
                Omega_sym[i, j] = Omega_tau[j, i]
    
    return Omega_sym

# Symmertrize all precision matrices
def symmetrize_all_precision_matrices(omega_tau_list):
    """
    Symmetrize all precision matrices Ω(tau) for each time point.

    Parameters:
    - omega_tau_list: List of (d × d) precision matrices Ω(tau) for each time point.

    Returns:
    - omega_sym_list: List of (d × d) symmetrized precision matrices.
    """
    omega_sym_list = [symmetrize_precision_matrix(Omega_tau) for Omega_tau in omega_tau_list]
    return omega_sym_list

def GIC_i(λ2, X, Xt_1, alpha_hat, tau_t_values, i, gamma_n_d, h):
    n, d = X.shape
    
    # Sum of squared errors
    residuals = []
    for t in range(1, n):
        xt_i = X[t, i]
        Xt_1_t = Xt_1[t-1, :]
        predicted = alpha_hat[i, :] @ Xt_1_t
        residuals.append(np.linalg.norm(xt_i - predicted) ** 2)
    
    sum_squared_error = np.sum(residuals) / n
    
    # Number of selected coefficients (si(λ2))
    s_i_lambda2 = np.count_nonzero(alpha_hat[i, :])
    
    # GIC formula
    GIC_value = np.log(sum_squared_error) + (gamma_n_d / n) * (36 * s_i_lambda2) / (35 * h)
    
    return GIC_value

def find_optimal_lambda2(X, Xt_1, alpha_hat, tau_t_values, i, gamma_n_d, h, λ2_initial):
    """
    This function finds the optimal λ2 by minimizing GIC_i.
    
    Parameters:
    - X: (n × d) matrix of observations.
    - Xt_1: (n-1 × d) matrix of lagged predictors.
    - alpha_hat: Estimated alpha coefficients from the LASSO.
    - tau_t_values: Array of time points (normalized).
    - i: Index for which GIC is computed.
    - gamma: Scalar for the GIC penalty term.
    - h: Bandwidth parameter.
    - λ2_initial: Initial guess for λ2.

    Returns:
    - Optimal λ2 that minimizes the GIC value.
    """
    # Define the objective function to minimize GIC_i(λ2)
    def objective(λ2):
        return GIC_i(λ2, X, Xt_1, alpha_hat, tau_t_values, i, gamma_n_d, h)
    
    # Use scipy's minimize function to find the λ2 that minimizes GIC
    result = minimize(objective, λ2_initial, bounds=[(1e-6, None)], method='L-BFGS-B')
    
    # Return the optimal lambda2
    return result.x[0], result.fun  # Optimal λ2 and the minimized GIC value


# First step
def firstStep(X, λ_values, tau_t_values, h):
    # Create lagged matrix Xt-1
    Xt_1 = create_lagged_matrix(X)
    
    # Initialize lists to store results for each time point tau_t
    preliminary_a = []
    preliminary_b = []
    optimal_lambda_1_results = []

    # Loop over each time point tau
    for tau in tau_t_values:
        best_BIC = np.inf
        best_alpha_hat, best_beta_hat = None, None
        best_lambda = None
        
        # Perform grid search over λ values
        for λ1 in λ_values:
            alpha_hat, beta_hat = local_linear_lasso(X, Xt_1, tau, tau_t_values, h, λ1)
            BIC_value = BIC_i(λ1, alpha_hat, beta_hat, tau, tau_t_values, X, Xt_1, h)
            
            print(BIC_value)
            if BIC_value < best_BIC:
                best_BIC = BIC_value
                best_alpha_hat = alpha_hat
                best_beta_hat = beta_hat
                best_lambda = λ1
                
        # Now use the best estimates from the grid search for optimization
        # Use bounds to ensure the optimization searches for a better λ1
        bounds = [(0.01, 1)]  # Adjust bounds if needed
        result = minimize(BIC_i, best_lambda, args=(best_alpha_hat, best_beta_hat, tau, tau_t_values, X, Xt_1, h), bounds=bounds)
            
        optimal_lambda_1 = result.x[0]  # Get the optimized λ1
        print(optimal_lambda_1)
        sys.exit()
        
        # Run the local linear LASSO estimation for the current tau using the optimal λ1
        alpha, beta = local_linear_lasso(X, Xt_1, tau, tau_t_values, h, optimal_lambda_1)
    
        # Store the results
        preliminary_a.append(alpha)
        preliminary_b.append(beta)
        optimal_lambda_1_results.append(optimal_lambda_1)

    # Convert results into arrays for better readability
    preliminary_a = np.array(preliminary_a)
    preliminary_b = np.array(preliminary_b)
    optimal_lambda_1_results = np.array(optimal_lambda_1_results)
    
    # Output the estimated alpha, beta, and optimal λ1 for each time point
    print(f"Optimal λ1 values: {optimal_lambda_1_results}")
    
    return preliminary_a, preliminary_b, optimal_lambda_1_results, Xt_1

# Second step
def secondStep(preliminary_a, h, n, d, X, Xt_1, alpha_hat, tau_t_values, preliminary_b, a0):
    # Compute D_hat
    D_hat = compute_D_hat(preliminary_a)
    print("D_hat values:\n", D_hat)
    
    λ2 = 0.1 # dit fixen 
    gamma = 0.1 # For empirical study, or 1 for simulation ZIE APPENDIX voor parameter tuning
    gamma_n_d = gamma * np.log(np.log(n)) * np.log(36 * d / (35 * h))
    
    λ2_initial = 0.1 # dit fixen 
    i = 1
    optimal_lambda2, minimized_GIC_value = find_optimal_lambda2(X, Xt_1, alpha_hat, tau_t_values, i, gamma, h, λ2_initial)
    
    # Compute the penalized objective function and get estimated A and B
    Q_i, alpha_estimates, beta_estimates = compute_penalized_objective(
        preliminary_a, preliminary_b, D_hat, X, Xt_1, tau_t_values, λ2, a0, h)
    
    # Output the final penalized objective value
    print("Penalized Objective Q_i:", Q_i)
    
    # Print the estimated transition matrices A (alpha) and B (beta)
    print_estimated_matrices(alpha_estimates, beta_estimates, tau_t_values)
    
    return alpha_estimates, beta_estimates

# Third step
def thirdStep(X, alpha_estimates, tau_t_values, bandwidth):
    residuals = estimate_residuals_with_estimated_A(X, alpha_estimates, p=1)

    # Print the shape and first few residuals for inspection
    print("Shape of residuals:", residuals.shape)
    print("First few residuals:\n", residuals[:5])

    sigma_tau_list = estimate_sigma_tau(residuals, tau_t_values, bandwidth)

    # Print the first time-varying covariance matrix for inspection
    print("First time-varying covariance matrix Σ(tau) at tau = 0:\n", sigma_tau_list[0])

    lambda_3 = 0.1  # You can adjust this value
    omega_tau_list = estimate_precision_matrices(sigma_tau_list, lambda_3)

    # Print the first time-varying precision matrix for inspection
    print("First time-varying precision matrix Ω(tau) at tau = 0:\n", omega_tau_list[0])        

    # Symmetrize the precision matrices
    omega_sym_list = symmetrize_all_precision_matrices(omega_tau_list)

    # Print the first symmetrized precision matrix for inspection
    print("First symmetrized precision matrix Ω(tau) at tau = 0:\n", omega_sym_list[0])

# Main
def main():

    # Define parameters
    n = 10  # Number of observations
    d = 20  # Number of variables
    tau_t_values = np.linspace(0, 1, n) # Create normalized time points tau_t
    h = 0.75 * (np.log(d) / n) ** (1/5) # Bandwidth for kernel as in Li, Ke and Zhang (2015)
    bandwidth = h  # also as in Li, Ke and Zhang (2015).
    λ_values = np.linspace(0.1, 1, 10)  # Range of λ values for grid search
    a0 = 3.7 # as suggested in Fan and Li (2001)

    # Generate example data
    X = np.random.randn(n, d)  # Random data as placeholder (replace with real data)

    # Use real data and n,d are the shape of the data, X is the data

    '''
    3.1: Preliminary time-varying LASSO estimation
    '''
    preliminary_a, preliminary_b, optimal_lambda_1_results, Xt_1 = firstStep(X, λ_values, tau_t_values, h)
    
    '''
    3.2: Penalised local linear estimation with weighted group LASSO
    '''
    alpha_estimates, beta_estimates = secondStep(preliminary_a, h, n, d, X, Xt_1, alpha_hat, tau_t_values, preliminary_b, a0)

    '''
    3.3: Estimation of the time-varying precision matri
    '''
    thirdStep(X, alpha_estimates, tau_t_values, bandwidth)


###########################################################
### Call main
if __name__ == "__main__":
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    print("\nExecution took:", timedelta(seconds=end_time - start_time))