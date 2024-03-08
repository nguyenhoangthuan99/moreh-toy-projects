make clean && make &&
# srun -p EM ./kmeans_seq data_0.5M_64_centroids/centroids.pt data_0.5M_64_centroids/data.pt result_seq.pt final_centroids.pt 1024 &&
# srun -p EM --gres=gpu:1  ./kmeans_gpu data_0.5M_64_centroids/centroids.pt data_0.5M_64_centroids/data.pt result_gpu.pt final_centroids_gpu.pt 1024 &&
# diff result_seq.pt result_gpu.pt 
# srun -p EM ./plot_data.py result final_centroids.pt data_0.5M_64_centroids/data.pt result_seq.pt output_seq.png &&
# srun -p EM ./plot_data.py result final_centroids_gpu.pt data_0.5M_64_centroids/data.pt result_gpu.pt output_gpu.png
# rocprof --stats --hip-trace


# srun -p EM ./kmeans_seq example_data/centroids.pt example_data/data.pt result_seq.pt final_centroids.pt 1024 &&
# srun -p EM --gres=gpu:1 --nodelist=em05  ./kmeans_gpu example_data/centroids.pt example_data/data.pt result_gpu.pt final_centroids_gpu.pt 1024 &&
# diff result_seq.pt result_gpu.pt 

# srun -p EM ./kmeans_seq data_valid/centroids.pt data_valid/data.pt result_seq.pt final_centroids.pt 1024 &&
# srun -p EM --gres=gpu:2 --nodelist=em05  ./kmeans_gpu data_valid/centroids.pt data_valid/data.pt result_gpu.pt final_centroids_gpu.pt 1024 &&

# srun -p EM ./kmeans_seq data_8M_128_centroids/centroids.pt data_8M_128_centroids/data.pt result_seq.pt final_centroids.pt 1024 &&
srun -p EM --gres=gpu:1 ./kmeans_gpu data_8M_128_centroids/centroids.pt data_8M_128_centroids/data.pt result_gpu.pt final_centroids_gpu.pt 1024 &&

# srun -p EM ./kmeans_seq data_8M_128_centroids/centroids.pt data_8M_128_centroids/data.pt result_seq.pt final_centroids.pt 1 &&
# srun -p EM --gres=gpu:1 ./kmeans_gpu data_8M_1024_centroids/centroids.pt data_8M_1024_centroids/data.pt result_gpu.pt final_centroids_gpu.pt 1024 &&

# srun -p EM --gres=gpu:2 ./kmeans_gpu data_10M_128_centroids/centroids.pt data_8M_128_centroids/data.pt result_gpu.pt final_centroids_gpu.pt 1024 &&
diff result_seq.pt result_gpu.pt 