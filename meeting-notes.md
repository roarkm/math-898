# 2/9/2022

TODO: Refine FFNN (1-2 layers) and dataset(s) to test verification and analysis.
TODO: Train, test, view dataset, dump weights, write explicit forumula, try verification.

## New questions
- How to translate a feedforward NN verification problem into an optimization problem?
  What is the overall picture of how this works?

- How does the Integer Linear Program (ILP) algorithm for verification work?

- How does structure (e.g. banded structure, sparsity, PSD)
  in the layer weight matrices affect verification?

# 2/16/2022
- Reviewed how to translate NN Verification into an ILP problem.
  Continue making this understanding robust.

- In the NNV encoding for ILP, what is X? Y? (i.e. a polytope?)
