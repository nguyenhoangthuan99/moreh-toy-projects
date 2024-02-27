#/bin/bash

make clean && make &&
srun -p EM --gres=gpu:2 ./edge test.jpg res.jpg res_hip.jpg 0