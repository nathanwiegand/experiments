# Cosine Similarity experiments
The results are from runing on Ubuntu 22.04.5 LTS with the following hardware:
- AMD Ryzen 9 7950X 16-Core Processor
- 192GB DDR5 RAM 4800MT/s
- NVIDIA GeForce RTX 4090 

## Commentary
Seems that both numpy and pytorch are faster than the C++ code that was produced.

Numpy is 10x faster than the SIMD code. We can cut the lead to 5x using BLAS.

Vibe coded all of the Python and C++ code using Claude.

## Results
Results from Python
```
CPU time: 0.2172 seconds
Result shape: (100000000,)
Total number of similarity scores: 100000000
---------------
GPU time: 0.2024 seconds
GPU speedup: 1.07x
Result shape: (100000000,)
Total number of similarity scores: 100000000
```

Results from C++
```
Running AVX similarity
Creating test data...
Calculating similarities...
Computation time: 1.442 seconds
Number of results: 100000000
----------------------------------------
Running aligned similarity
Creating test data...
Calculating similarities...
Computation time: 1.00 seconds
Number of results: 100000000
----------------------------------------
Running BLAS similarity
Creating test data...
Calculating similarities...
Computation time: 1.096 seconds
Example results (first 5):
Number of results: 100000000
----------------------------------------
Running SIMD similarity
Creating test data...
Calculating similarities...
Computation time: 2.273 seconds
Results size: 100000000
----------------------------------------
```