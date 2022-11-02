## [Low-rank lottery tickets: finding efficient low-rank neural networks via matrix differential equations](https://arxiv.org/abs/2205.13571)

Code supplement for all dense neural network experiments of the arXiv Preprint

### Usage

1. create a python virtual environment (pyenv or conda). If you are using no virtual environment, please be aware of
   version incompatibilities of tensorflow
2. Install the project requirements (example for pip):
   ``pip install -r requirements.txt``
3. Run the batch scripts for the test cases
    1. ``sh run_tests_dense_reference.sh`` trains a baseline traditional dense network.
    2. ``sh run_tests_fixed_rank.sh`` trains a fixed low-rank network using our proposed method. This method is used in
       section 5.1
    3. ``sh run_tests_rank_adaptive.sh`` trains a network using our proposed rank adaptive algorithm to find optimal low
       ranks
    4. ``sh run_tests_fixed_rank_fine_tuning.sh`` trains a network using our proposed rank adaptive algorithm to find
       optimal low ranks. Once the low-ranks are found, the script switches to dynamical fixed rank training to
       fine-tune the model. This method is used in Section 5.2
    4. ``sh run_tests_fixed_rank_train_from_prune.sh`` loads the weights of a traditional network (provided in the
       folder "dense_weights"), then factorizes the weight matrix and truncates all but 20 eigenvalues. Then, fixed
       low-rank training is used to retrain the model. This method is used in Section 7.3
       
### Useful links

The pytorch version can be found [here](https://github.com/emabuell/pytorch_dlr](https://github.com/COMPiLELab/DLRT/tree/efficient_gradient)
