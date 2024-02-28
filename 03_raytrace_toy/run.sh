#/bin/bash

make clean && srun -p EM make &&
srun -p EM --gres=gpu:2 ./raytracer 1024 1024 0 test.jpg 