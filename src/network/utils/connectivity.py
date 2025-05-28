import numpy as np


def create_matrix_with_oscillations(n_nodes, M_oscillatory_components, oscillation_decay_rate=0.1, osc_frequencies_b = None, real_eigenvalues = None):
    """
    Creates a connectivity-style matrix with specified decaying and damped oscillatory modes.

    The initial matrix A_intermediate = Q @ D_matrix @ Q.T has eigenvalues
    defined by D_matrix. Setting the diagonal of A to zero afterwards will
    alter these eigenvalues.

    Args:
        n_nodes (int): Total number of nodes in the matrix.
        M_oscillatory_components (int): Number of oscillatory components.
                                         Each component corresponds to a pair of
                                         complex conjugate eigenvalues (a +/- bi),
                                         where 'a' is negative for decay.
                                         This means 2 * M_oscillatory_components
                                         eigenvalues will be dedicated to oscillations.
        oscillation_decay_rate (float): The magnitude of the negative real part 'a'
                                        for the oscillatory eigenvalues (e.g., 0.1 means 'a' is -0.1).
                                        Must be positive to ensure decay.

    Returns:
        numpy.ndarray: The generated n_nodes x n_nodes matrix A, with zero diagonal.
        numpy.ndarray: The orthogonal matrix Q whose columns form the basis.
        numpy.ndarray: The block-diagonal matrix D_matrix that defined the initial
                       eigenvalue structure before diagonal zeroing.
    """
    if not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer.")
    if not isinstance(M_oscillatory_components, int) or M_oscillatory_components < 0:
        raise ValueError("M_oscillatory_components must be a non-negative integer.")
    if 2 * M_oscillatory_components > n_nodes:
        raise ValueError(
            "Total number of nodes must be at least twice the "
            "number of oscillatory components (each requires two dimensions)."
        )
    if not isinstance(oscillation_decay_rate, (int, float)) or oscillation_decay_rate <= 0:
        raise ValueError("oscillation_decay_rate must be a positive number for decay.")

    # 1. Create an arbitrary orthogonal basis Q
    # Q's columns will effectively be the eigenvectors (or span invariant subspaces
    # for complex eigenvalues) for the matrix A_intermediate = Q @ D_matrix @ Q.T
    Q_random_matrix = np.random.randn(n_nodes, n_nodes)
    Q, _ = np.linalg.qr(Q_random_matrix)  # Q is n_nodes x n_nodes, orthogonal

    # 2. Create the block-diagonal matrix D_matrix that defines the eigenvalue structure
    D_matrix = np.zeros((n_nodes, n_nodes))

    # --- Damped Oscillatory components (eigenvalues a +/- bi, where a < 0) ---
    # Each pair of eigenvalues requires a 2x2 block in D_matrix.
    # The 'b_val' are the frequencies of oscillation.
    # The 'a_val' is the real part, determining decay rate.
    a_val_osc = -abs(oscillation_decay_rate) # Ensure it's negative for decay

    if M_oscillatory_components > 0:
        if M_oscillatory_components == 1:
            # A single frequency for the oscillatory component
            if osc_frequencies_b is None:
                osc_frequencies_b = np.array([0.7]) # Example frequency (imaginary part b)
        else:
            # Spread frequencies for multiple oscillatory components
            if osc_frequencies_b is None:
                osc_frequencies_b = np.linspace(0.2, 0.8, M_oscillatory_components)

        for i in range(M_oscillatory_components):
            b_val = osc_frequencies_b[i]
            # For eigenvalues a_val_osc +/- i*b_val, the real 2x2 block is:
            #   [[a_val_osc,    b_val],
            #    [-b_val,    a_val_osc]]
            # This block is placed at the (2*i)-th and (2*i+1)-th rows/columns.
            start_idx = 2 * i
            D_matrix[start_idx, start_idx]     = a_val_osc
            D_matrix[start_idx, start_idx + 1] = b_val
            D_matrix[start_idx + 1, start_idx] = -b_val
            D_matrix[start_idx + 1, start_idx + 1] = a_val_osc

    # --- Real, purely decaying eigenvalues for the remaining dimensions ---
    num_real_eigenvalues = n_nodes - 2 * M_oscillatory_components
    if num_real_eigenvalues > 0:
        # These eigenvalues should be real and negative for decay.
        # Magnitudes < 1 for stability if discrete, negative for continuous time stability.
        # Example: -exp(-x) where x > 0.
        if real_eigenvalues is None:
            if num_real_eigenvalues == 1:
                real_eigenvalues = np.array([-0.5]) # Example for a single real eigenvalue
            else:
                # Values from approx -0.90 (slower decay) to -0.22 (faster decay)
                real_eigenvalues = -np.exp(-np.linspace(0.1, 1.5, num_real_eigenvalues))

        for i in range(num_real_eigenvalues):
            # Place these on the diagonal of D_matrix, after the oscillatory blocks.
            diag_idx = 2 * M_oscillatory_components + i
            D_matrix[diag_idx, diag_idx] = real_eigenvalues[i]


    # 3. Construct the intermediate matrix A_intermediate
    # The eigenvalues of A_intermediate are the eigenvalues of D_matrix.
    A = Q @ D_matrix @ Q.T


    return A, Q, D_matrix