# Loads a dense 5-layer neural networks with width [784,784,784,784,10]
# Then, the dense, full-rank weight matrices are factorized using SVD, and we keep the top 20 eigenvalue-eigenvector pairs.
# The decomposed network is first evaluated (first line of the history file), and then retrained using our fixed-rank training algorithm.
python src/mnist_DLRA_fixed_rank_retrain_from_prune.py -s 20 -l 1 --train 1
