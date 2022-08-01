# Fixed Rank training, for 5-layer network of widths [500,500,500,500,10] with adaptive low-ranks  for 10 epochs. Last layer has fixed rank 10 (since we classfy 10 classes)
# Starting rank is set to 150, rank adaption tolerance is set to 0.17, and max rank to 300.
python cifar10_Conv.py -s 150 -t 0.01 -l 0 -a 1
