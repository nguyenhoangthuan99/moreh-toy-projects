#/bin/bash

make clean && make &&
srun -p EM --gres=gpu:2 ./raytracer 2048 2048 1 test.jpg 