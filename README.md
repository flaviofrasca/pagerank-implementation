# PageRank Implementation

Implementation of the PageRank algorithm using the Power Method, tested on synthetic graphs and on the real Hollins University web graph dataset.

## Project goals
- Implement PageRank efficiently using sparse matrix logic
- Analyze convergence behavior
- Test the algorithm on synthetic and real-world web graphs

## Dataset
- hollins.dat
- 6012 pages
- 23875 links

## Repository structure
- `src/`: source code
- `data/`: dataset
- `report/`: final PDF report

## Main results
- Convergence in 47 iterations on the Hollins dataset
- Correct handling of 3189 dangling nodes
- Realistic top-10 ranking led by the university homepage

## How to run
```bash
python src/hollins_dataset_analysis.py
```

## Report
See `report/pagerank-report.pdf`