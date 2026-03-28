import numpy as np
from pathlib import Path

""" 
    Parameters:
    - A: link matrix (n x n) 
    - m: damping factor 
    - max_iter: maximum number of iterations
    - tol: tollerance for convergence 

    Returns:
    - v_tilde: Page Rank vector (sum to 1)
    - lambda_new: estimated dominant eigenvalue (Rayleigh quotient)
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
        lambda_new = np.dot(v, v_tilde) / np.dot(v,v)

        # Convergence Control:
        # |lambda^(m+1) - lambda^(m)| < tol * |lambda^(m+1)|
        # If the absolute difference is less then tol * |lambda^(m+1)|, STOP.
        if abs(lambda_new - lambda_old) < tol * abs(lambda_new):
            return v_tilde, lambda_new, iteration

        # Updating Variables for the next iteration:
        v = v_tilde
        lambda_old = lambda_new
        iteration += 1

    return v_tilde, lambda_new, iteration


'''
    DATASET'S PREPROCESSING

1) FORMAT PARSING:
  - The file starts with metadata (6012 pages, 23875 links)
  - Followed by a list URLs (e.g.: 1 http://...)
  - Followed by a list of edges/links in the format (ID_Sorgente ID_Destinazione)
  - It's necessary to separate the URLs from edges/links 

2) CONSTRUCTION OF THE COLUMN-STOCHASTIC MATRIX A:
  - The algorithm require that A[i][j] is the probability to go from j to i
  - If page j has k outgoing links then each link must have a weight of 1/k
  - We need to count how many outgoing links k each page j has and divide 1 by that number k

3) HANDLING "DANGLING NODES":
  - Problem: the web pages without outgoing links (Dangling Nodes) create zero columns 
    and this "breaks" column-stochasticity of the matrix.
  - Solution: if a page has no outgoing links, it is assumed that 
    the user clicks on a random page among all available ones.
  - If a column of the matrix is all zeros, fill it with 1/N (where N is the total number of pages)

'''

def load_data(filename):
    print(f"File reading {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return np.array([]), {}

    # 1) FORMAT PARSING:
    num_pages = int(lines[0].split()[0])
    
    urls = {}
    links = []

    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue

        # Let's check if the line is a URL
        if len(parts) >= 2 and ("http" in parts[1] or not parts[1].isdigit()):
            page_id = int(parts[0])
            url = parts[1]
            urls[page_id] = url
        else:
            # If the line isn't a URL then we treat it as a link 
            if len(parts) >= 2:
                src = int(parts[0])
                dst = int(parts[1])
                links.append((src, dst))

    print(f"Found {len(urls)} URL e {len(links)} links")

    # 2) CONSTRUCTION OF THE COLUMN-STOCHASTIC MATRIX A:
    A = np.zeros((num_pages, num_pages), dtype=float)

    # Counting how many outgoing links each web page has (out-degree)
    out_degree = np.zeros(num_pages + 1, dtype=int)  # 1-based indexing
    for src, dst in links:
        if src <= num_pages and dst <= num_pages:
            out_degree[src] += 1

    # Building the matrix A
    for src, dst in links:
        if src <= num_pages and dst <= num_pages:
            row = dst - 1       # 0-based based row index (destination page)
            col = src - 1       # 0-based based column index (source page)
            if out_degree[src] > 0:
                A[row, col] = 1.0 / out_degree[src]

    # 3) HANDLING "DANGLING NODES":
    dangling_count = 0
    uniform_prob = 1.0 / num_pages
    for j in range(num_pages):
        if out_degree[j + 1] == 0:
            dangling_count += 1
            A[:, j] = uniform_prob 

    print(f"Handled {dangling_count} 'Dangling Nodes'")
    return A, urls


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_PATH = BASE_DIR / "data" / "raw" / "hollins.dat"

    print(f"Dataset path: {DATASET_PATH}")
    print(f"Exists: {DATASET_PATH.exists()}")

    A, urls = load_data(str(DATASET_PATH))

    if (A.size > 0 and len(urls) > 0):
        m = 0.15
        maxIter = 100
        tol = 1e-6
        pr_vector, aut_val, iters = page_rank(A, m, maxIter, tol)
    else:
        exit("Error: file 'hollins.dat' not found or empty!")

    # Top 10 Ranking:
    # Finding the indexes of the pages with the highest scores
    top_indices = np.argsort(pr_vector)[::-1][:10]  # indexes of the top 10 pages (0-based)
    top_scores = pr_vector[top_indices]

    print("\nTop ten PageRank scores:")
    for i, idx in enumerate(top_indices):
        page_id = idx + 1  # converting to 1-based to access to urls 
        url = urls.get(page_id, "URL not found")
        print(f"{i+1}. Score: {top_scores[i]:.6f} - URL: {url}")

    print("\nEstimated dominant eigenvalue (Rayleigh quotient):", aut_val)
    print("Number of iterations:", iters)