#/bin/bash

make clean && make &&
srun -p EM --gres=gpu:2 ./edge Big_Tiger_Cub.jpg res.jpg res_hip.jpg 1