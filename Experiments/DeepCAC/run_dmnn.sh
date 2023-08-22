echo "DeepCAC" >> Results/results_DMNN.txt

# echo "DMNN" >> Results/results_DMNN.txt
# echo "CIC" >> Results/results_DMNN.txt

# for j in 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset cic --n_runs 5 --n_epochs 100 --alpha 0 --beta 0.0 --gamma 1 --eta 0.0 --sub_epochs False --data_ratio -1  --n_clusters $j --n_z 32 --expt DeepCAC >> Results/results_DMNN.txt
# done


# echo "DMNN" >> Results/results_DMNN.txt
# echo "Wid_Mortality" >> Results/results_DMNN.txt

# for j in 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset wid_mortality --n_runs 5 --n_epochs 100 --alpha 0 --beta 0.0 --gamma 1 --eta 0.0 --sub_epochs False --data_ratio -1  --n_clusters $j --n_z 32 --expt DeepCAC >> Results/results_DMNN.txt
# done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Diabetes" >> Results/results_DMNN.txt

# for j in 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset diabetes --n_runs 5 --n_epochs 100 --alpha 0 --beta 0.0 --gamma 1 --eta 0.0 --sub_epochs False --data_ratio -1  --n_clusters $j --n_z 32 --expt DeepCAC >> Results/results_DMNN.txt
# done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "CIC_LOS" >> Results/results_DMNN.txt

# for j in 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset cic_los --n_runs 5 --n_epochs 100 --alpha 0 --beta 0.0 --gamma 1 --eta 0.0 --sub_epochs False --data_ratio -1  --n_clusters $j --n_z 32 --expt DeepCAC  --n_classes 3 >> Results/results_DMNN.txt
# done

echo "DMNN" >> Results/results_DMNN.txt
echo "Sepsis24" >> Results/results_DMNN.txt

for j in 5 15 50
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../../dmnn_torch.py --dataset sepsis24 --n_runs 5 --n_epochs 100 --alpha 0 --beta 0.0 --gamma 1 --eta 0.0 --sub_epochs False --data_ratio -1  --n_clusters $j --n_z 32 --expt DeepCAC >> Results/results_DMNN.txt
done


# echo "DMNN" >> Results/results_DMNN.txt
# echo "Respiratory24" >> Results/results_DMNN.txt

# for j in 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset ards24 --n_runs 5 --n_epochs 100 --alpha 0 --beta 0.0 --gamma 1 --eta 0.0 --sub_epochs False --data_ratio -1  --n_clusters $j --n_z 32 --expt DeepCAC >> Results/results_DMNN.txt
# done