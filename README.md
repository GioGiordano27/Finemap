# Finemap
First , we implement efficient Bayes factor for each causal configurations. 
Second, we implement the prior calculation for each configurations. 
Third, we implement posterior inference over all possible configurations assuming at maximum 3 causal SNPs. Some configurations result in non-finite multivariate Gaussian density, which
we discarded.
Fourth, we implement posterior inclusion probabilities (PIP) to calculate SNP-level posterior probabilities.
