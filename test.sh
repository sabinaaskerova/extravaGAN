#!/bin/bash
if [ ! -f results_lsgan.txt ]; then
    touch results_lsgan.txt
fi
if [ ! -f checkpoints/G_0.05.pth ]; then
    touch checkpoints/G_0.05.pth
fi
if [ ! -f checkpoints/D_0.05.pth ]; then
    touch checkpoints/D_0.05.pth
fi

python3 train.py --noise_factor 0.05 --gname "G_0.05.pth" --dname "D_0.05.pth"
python3 generate.py --gname "G_0.05.pth"
echo "--noise_factor 0.05:" >> results_lsgan.txt
python -m pytorch_fid samples test_images >> results_lsgan.txt
python improved_precision_recall.py test_images samples --num_samples 10000 >> results_lsgan.txt

if [ ! -f checkpoints/G_0.01.pth ]; then
    touch checkpoints/G_0.01.pth
fi
if [ ! -f checkpoints/D_0.01.pth ]; then
    touch checkpoints/D_0.01.pth
fi

python3 train.py --noise_factor 0.01 --gname "G_0.01.pth" --dname "D_0.01.pth"
python3 generate.py --gname "G_0.01.pth"

echo "--noise_factor 0.01:" >> results_lsgan.txt
python -m pytorch_fid samples test_images >> results_lsgan.txt
python improved_precision_recall.py test_images samples --num_samples 10000 >> results_lsgan.txt