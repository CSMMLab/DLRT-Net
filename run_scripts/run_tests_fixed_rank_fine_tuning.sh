# Fixed Rank training, for 5-layer network of widths [500,500,500,500,10] wit adpative low-ranks  for 10 epochs. Last layer has fixed rank 10 (since we classfy 10 classes)
# Starting rank is set to 300, rank adaption tolerance is set to 0.17
python src/mnist_DLRA.py -s 300 -t 0.17 -l 0 -a 1 -d 500
# Fixed Rank finetuning for 100 epochs (flags -s and -t are set only to navigate into the right save-directory)
python src/mnist_DLRA_fixed_rank.py -s 300  -t 0.17 -l 1 --train 1 -d 500
