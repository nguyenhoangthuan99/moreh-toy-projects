#/bin/bash

make clean && make &&
srun -p EM --gres=gpu:1 ./raytracer  49152 49152 0 #test.jpg 