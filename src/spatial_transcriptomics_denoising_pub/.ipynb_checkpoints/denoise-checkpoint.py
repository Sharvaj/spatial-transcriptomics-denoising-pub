
import numpy as np
import scipy
from scipy.special import xlogy

from . import utilities



def krr_laplacian_oneshot(training_spatial_locations, training_targets, kernel_obj, effective_Laplacian, lam_krr, omega_lap=0, kernel_mat=None):
    """
    Performs one-shot Kernel Ridge Regression with a Laplacian regularizer.

    This function solves the system of linear equations to find the optimal
    weights for a Kernel Ridge Regression model that incorporates a Laplacian
    regularizer.

    Args:
        training_spatial_locations (np.ndarray): Array of spatial coordinates for training data. Shape: (m_training, 2).
        training_targets (np.ndarray): Array of target values (e.g., gene expression) for training. Shape: (m_training, d).
        kernel_obj (callable): A function that computes the kernel matrix.
        effective_Laplacian (np.ndarray): The effective Laplacian matrix for the spatial graph. Shape: (m_training, m_training).
        lam_krr (float): The regularization parameter for the standard KRR term.
        omega_lap (float): The regularization parameter for the Laplacian term. Defaults to 0.
        kernel_mat (np.ndarray, optional): Pre-computed kernel matrix for efficiency. Shape: (m_training, m_training).
            If None, the function will compute it.

    Returns:
        dict: A dictionary containing:
            - "theta_hat" (np.ndarray): The optimal weights vector. Shape: (m_training, d).
            - "F_hat_unnormalized_fun" (callable): A function to predict on new data.
    """
    m_training = training_spatial_locations.shape[0] # num_samples
    if kernel_mat is None:
        # Calculate the kernel matrix if not provided (note the correct normalization)
        kernel_mat = kernel_obj(training_spatial_locations) / m_training

    # Construct the system matrix for the linear solver
    system_matrix = lam_krr * np.eye(m_training) + kernel_mat + omega_lap * (effective_Laplacian @ kernel_mat)

    # Solve for the weights (theta_hat) using numpy's linear solver
    theta_hat = np.linalg.solve(kernel_mat @ system_matrix, kernel_mat @ training_targets) / np.sqrt(m_training)

    # Solution function
    def F_hat_unnormalized_fun(query_spatial_locations):
        return (kernel_obj(query_spatial_locations, training_spatial_locations) @ theta_hat) / np.sqrt(m_training)
    
    return {"theta_hat": theta_hat,
            "F_hat_unnormalized_fun": F_hat_unnormalized_fun}


def set_good_parameters_using_eigs_and_reads(training_spatial_locations, training_targets, kernel_obj, effective_Laplacian, 
                                             training_reads, omega_rel_default=6.0, fit_est_factor=0.7, 
                                             bisec_lb=1e-5, bisec_ub=1e+2, relative_buffer=1e-2, dist_reltol=1e-2):
    """
    Finds appropriate regularization strength for KRR using a bisection search.

    This function determines appropriate regularization parameters (lam_krr and omega_lap)
    by performing a bisection search to match a desired fit estimate, which is
    derived from the expected noise level based on sequencing reads.

    Args:
        training_spatial_locations (np.ndarray): Spatial coordinates for training data. Shape: (m_training, 2).
        training_targets (np.ndarray): Target values for training. Shape: (m_training, d).
        kernel_obj (callable): A function to compute the kernel matrix.
        effective_Laplacian (np.ndarray): The effective Laplacian matrix. Shape: (m_training, m_training).
        training_reads (np.ndarray): Array of read counts for each spot/pixel, used for noise estimation. Shape: (m_training,).
        omega_rel_default (float, optional): Default relative weight for the Laplacian regularizer.
        fit_est_factor (float, optional): Factor to scale the estimated noise level to get the intended fit estimate.
        bisec_lb (float, optional): Lower bound for the bisection search.
        bisec_ub (float, optional): Upper bound for the bisection search.
        relative_buffer (float, optional): Tolerance for the bisection search.
        dist_reltol (float, optional): Relative tolerance for checking bisection convergence.

    Returns:
        dict: A dictionary with the determined parameters and final fit estimate:
            - "lam_krr" (float): The final KRR regularization parameter.
            - "omega_lap" (float): The final Laplacian regularization parameter.
            - "overall_strength_factor" (float): The overall scaling factor found by the bisection search.
            - "fit_estimate" (float): The final mean squared error.
    """
    m_training = training_spatial_locations.shape[0]
    kernel_mat = kernel_obj(training_spatial_locations) / m_training

    # Get the largest eigenvalue of the kernel matrix to set a default scaling factor
    my_res = scipy.sparse.linalg.eigsh(kernel_mat, k=1, which='LM', return_eigenvectors=False)
    lam_krr_rel_default = my_res[0]

    # Estimate the noise level from the training reads
    noise_estimate = np.mean(np.minimum(1 / training_reads, 1))
    intended_fit_estimate = fit_est_factor * noise_estimate

    # set up the bisection
    print("Bisection search for overall regularization strength ---------------------------")

    success_flag = False
    while not success_flag:
        overall_strength_iterate = (bisec_lb + bisec_ub) / 2
        sc1 = (lam_krr_rel_default) * overall_strength_iterate
        sc2 = (omega_rel_default) * overall_strength_iterate

        # Check for convergence based on relative distance
        if ((bisec_ub - bisec_lb) / overall_strength_iterate < dist_reltol):
            print('Converged but success criterion not necessarily achieved!')
            break

        # Run the KRR model with the current parameters
        spatial_results = krr_laplacian_oneshot(training_spatial_locations, training_targets, kernel_obj, effective_Laplacian, lam_krr=sc1, omega_lap=sc2, kernel_mat=kernel_mat)
        
        # Normalize the predictions and calculate the fit estimate (MSE)
        # Note: This line assumes a `utilities` module exists.
        estimate_spatial = utilities.normalize_matrix(np.maximum(spatial_results["F_hat_unnormalized_fun"](training_spatial_locations), 0), axis=1) # NOT transposed !!
        fit_estimate = (1 / m_training) * np.linalg.norm(training_targets - estimate_spatial)**2

        print(f"Ratio = {(fit_estimate / intended_fit_estimate):.3f}")

        # Adjust the bisection bounds based on the fit estimate
        if fit_estimate > (1 + relative_buffer) * intended_fit_estimate: # regularization too strong
             bisec_ub = overall_strength_iterate
        elif fit_estimate < (1 - relative_buffer) * intended_fit_estimate: # regularization too weak
             bisec_lb = overall_strength_iterate
        else:
             success_flag = True
             print('Converged and success criterion achieved!')

    print("-------------------------------------------------------------------------------")

    return {"lam_krr": sc1, "omega_lap": sc2, "overall_strength_factor": overall_strength_iterate, "fit_estimate": fit_estimate}



