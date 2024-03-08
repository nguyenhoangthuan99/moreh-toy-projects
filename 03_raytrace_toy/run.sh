#/bin/bash

make clean && make &&
srun -p EM --gres=gpu:2 ./raytracer  32768 32768 0 # test.jpg 