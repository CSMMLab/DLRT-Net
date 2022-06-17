# Fixed Rank training, for 5-layer network of widths [500,500,500,500,10] with adaptive low-ranks  for 10 epochs. Last layer has fixed rank 10 (since we classfy 10 classes)
# Starting rank is set to 300, rank adaption tolerance is set to 0.17
python mnist_DLRA.py -s 300 -t 0.17 -l 0 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.17 -l 1 -a 1 -d 500

python mnist_DLRA.py -s 300 -t 0.15 -l 0 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.15 -l 1 -a 1 -d 500

python mnist_DLRA.py -s 300 -t 0.05 -l 0 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500
python mnist_DLRA.py -s 300 -t 0.05 -l 1 -a 1 -d 500