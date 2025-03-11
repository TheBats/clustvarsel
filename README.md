# clustvarsel
This repo contains an implementation of the Clustering Variable Selection algorithm (1) in Rust.

**Disclaimer** It is still being worked on and needs many improvement for it to be actually useful. Any feedback is welcome.
## Goal
The primary goal of this project is to learn Rust by implementing the algorithm from scratch. 
Thanks to this project I familiarized myself with the language as well as the different tools it comes with, such as benchmarking and testing. 
It also allowed me to apply what I learned in my Algorithms and Machine Learning classes. Lastly, because the algorithm is used on huge datasets, 
I had to keep memory and cpu use in mind when developing it and I spent a lot of time trying to find the best approaches.

## To improve
These are the various tasks I need to work on:
- [ ] Multithreading the `add` and `remove` steps in the ClustVarSel algorithm
- [ ] Writing a proper documentation
- [ ] Divide and Conquer approach for the matrix multiplication

# References
(1): Raftery, A. E. and Dean, N. (2006) Variable Selection for Model-Based Clustering. Journal of the American Statistical Association, 101(473), 168-178.
