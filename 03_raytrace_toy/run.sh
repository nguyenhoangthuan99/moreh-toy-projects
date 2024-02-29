#/bin/bash

make clean && srun -p EM make &&
srun -p EM --exclusive --gres=gpu:1 ./raytracer 32768 32768 0 #test.jpg 