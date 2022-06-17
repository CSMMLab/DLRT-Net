# Fixed Rank training, for 5-layer network of widths [500,500,500,500,10] wit low-ranks [20,20,20,20,10] for 100 epochs. Last layer has fixed rank 10 (since we classfy 10 classes)
#python mnist_DLRA_fixed_rank.py -s 10 -t 1.0 -l 0 --train 1 -d 500
#python mnist_DLRA_fixed_rank.py -s 20 -t 1.0 -l 0 --train 1 -d 500
#python mnist_DLRA_fixed_rank.py -s 30 -t 1.0 -l 0 --train 1 -d 500
#python mnist_DLRA_fixed_rank.py -s 40 -t 1.0 -l 0 --train 1 -d 500
#python mnist_DLRA_fixed_rank.py -s 50 -t 1.0 -l 0 --train 1 -d 500
python mnist_DLRA_fixed_rank.py -s 60 -t 1.0 -l 0 --train 1 -d 500
python mnist_DLRA_fixed_rank.py -s 70 -t 1.0 -l 0 --train 1 -d 500
python mnist_DLRA_fixed_rank.py -s 80 -t 1.0 -l 0 --train 1 -d 500
python mnist_DLRA_fixed_rank.py -s 90 -t 1.0 -l 0 --train 1 -d 500
python mnist_DLRA_fixed_rank.py -s 100 -t 1.0 -l 0 --train 1 -d 500