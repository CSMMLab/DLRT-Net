# Fixed Rank training, for 5-layer network of widths [500,500,500,500,10] with adaptive low-ranks  for 10 epochs. Last layer has fixed rank 10 (since we classfy 10 classes)
# Starting rank is set to 150, rank adaption tolerance is set to 0.17, and max rank to 300.
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.17 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.15 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.1 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.05 -l 0 -a 1 -d 784 -m 300 -e 250

python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.17 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.15 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.1 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.05 -l 0 -a 1 -d 784 -m 300 -e 250

python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.17 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.15 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.1 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.05 -l 0 -a 1 -d 784 -m 300 -e 250

python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.17 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.15 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.1 -l 0 -a 1 -d 784 -m 300 -e 250
python src/mnist_DLRT_parallel_integrator.py -s 150 -t 0.05 -l 0 -a 1 -d 784 -m 300 -e 250
