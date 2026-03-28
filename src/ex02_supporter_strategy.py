import numpy as np

""" 
    Parameters:
    - A: link matrix (n x n) 
    - m: damping factor 
    - max_iter: maximum number of iterations
    - tol: tollerance for convergence 

    Returns:
    - v: Page Rank vector normalized (sum to 1)
    - lambda_dom: estimated dominant eigenvalue
    - iteration: number of iterations perfomed 
"""

def page_rank(A, m, max_iter, tol):

    lambda_old = np.inf
    iteration = 0
    n = A.shape[0] 
    s = np.ones(n)
    
    # Inizialization of Page Rank vector:
    v = np.ones(n) / n

    # Power Method:   
    while iteration < max_iter:
        v_tilde = (1-m) * A @ v + m/n * s 

        # Rayleigh Quotient:
        lambda_new = np.dot(v, v_tilde) / np.dot(v, v)

        # Convergence Control:
        # |lambda^(m+1) - lambda^(m)| < tol * |lambda^(m+1)|
        # If the absolute difference is less then tol * |lambda^(m+1)|, STOP.
        if abs(lambda_new - lambda_old) < tol * abs(lambda_new):
            return v_tilde, lambda_new, iteration

        # Updating Variables for the next iteration:
        v = v_tilde
        lambda_old = lambda_new
        iteration += 1

    return v_tilde, lambda_new, max_iter


if __name__ == "__main__":
    A = np.array([[0,0,1/2,1/2,0],
             [1/3,0,0,0,0],
             [1/3,1/2,0,1/2,1],
             [1/3,1/2,0,0,0],
             [0,0,1/2,0,0]])
    m=0
    maxIter=100
    tol=1e-12
    pr_vector, eig_value, iters = page_rank(A, m, maxIter, tol)

    print("PageRank vector:", pr_vector)
    print("Estimated dominant eigenvalue (Rayleigh quotient):", eig_value)
    print("Number of iterations:", iters)

