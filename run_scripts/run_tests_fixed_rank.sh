# Fixed Rank training, for 5-layer network of widths [500,500,500,500,10] wit low-ranks [20,20,20,20,10] for 100 epochs. Last layer has fixed rank 10 (since we classfy 10 classes)
python src/mnist_DLRt_fr.py -s 20 -t 1.0 -l 0 --train 1 -d 500