# TODO list for CUDA Neural Network

1. (DONE) Fix random number generator for weight initialization
2. (DONE) Parallelize node and weight initialization
  * parallelize by layer
  * These are not order dependent
3. (DONE) Fix method 2 of feed forward
4. (DONE) Write a backprop method similar to method 2 of feed forward
5. (DONE) Modify output_error_gradients kernel to handle multiple threads dimensions and blocks
  * Also modified the remaining backprop kernels besides calc hidden err gradient, which relies on a linear combination and reduction
6. (DONE) Parallelize mse and accuracy calcs or maintain state
7. (DONE – Updated to run in separate stream) Update mse_sum kernel to use optimized reduction
8. Transfer patterns during compute stage
9. (TRIED – CURRENTLY SLOW) Optimize reduction for feed forward and backprop
  * calculate one level of reduction in calculation of feed forward / backprop
10. Profile feed forward and backprop
11. Split onto 2 GPUs
  * Run tests with random datasets to find ideal split
12. Split onto n GPUs
13. (DONE) Transfer dataset incrementally to GPU if larger than 2GB
14. (DONE) Add function to read net from file
15. Run a test case
16. Compare to FANN
17. Implement RProp
18. Ensure that weight initialization with curand does not have race condition.
  * Should be ok since only 32 threads can run at a single time. But if there are more than 1 sm processors, then the curand states could be accessed simultaneously.
19. (DONE) Add batch learning option
20. Modify for arbitrary number of layers
21. Calculate optimal block and grid sizes -- See [here](http://stackoverflow.com/questions/5810447/cuda-block-and-grid-size-efficiencies) to calculate params for kernels. Make threads a class field in some way so not defined as 128 in every function.
22. Add feature selection to dataset processor
23. (DONE) Modify cuda_ann for command line input
24. (DONE) Analyze kernels for memory coalescing. Change array indexing if necessary.
  * This improved the time for one epoch on the face dataset (6977 2960 1) from 284 seconds to epoch to less than 57.
  * Try using shared mem to access that require indexing in opposite manner

