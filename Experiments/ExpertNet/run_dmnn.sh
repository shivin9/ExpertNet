# echo "DMNN" >> Results/results_DMNN.txt
# echo "Kidney New" >> Results/results_DMNN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset aki_new --n_clusters $j --n_runs 5 --n_epochs 100 --n_z 32  >> Results/results_DMNN.txt
# done


# echo "DMNN" >> Results/results_DMNN.txt
# echo "Respiratory New" >> Results/results_DMNN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset ards_new --n_clusters $j --n_runs 5 --n_epochs 100 --n_z 32  >> Results/results_DMNN.txt
# done


# echo "DMNN" >> Results/results_DMNN.txt
# echo "CIC New" >> Results/results_DMNN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset cic_new --n_clusters $j --n_runs 5 --n_epochs 100 --n_z 32  >> Results/results_DMNN.txt
# done


# echo "DMNN" >> Results/results_DMNN.txt
# echo "CIC-LoS New" >> Results/results_DMNN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset cic_los_new --n_clusters $j --n_runs 5 --n_epochs 100 --n_z 32 --n_classes 3 >> Results/results_DMNN.txt
# done


echo "DMNN Experiments" >> Results/results_DMNN.txt

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Kidney 48" >> Results/results_DMNN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset aki48 --n_clusters $j --n_runs 5 --n_epochs 100 --n_z 32 >> Results/results_DMNN.txt
# done

echo "DMNN" >> Results/results_DMNN.txt
echo "Respiratory 24" >> Results/results_DMNN.txt

for j in 2 3 4 5
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../../dmnn_torch.py --dataset ards24 --n_clusters $j --n_runs 21 --n_epochs 100 --n_z 32 --eta 25 --sub_epochs False --data_ratio -1 >> Results/results_DMNN.txt
done


echo "DMNN" >> Results/results_DMNN.txt
echo "Sepsis 24" >> Results/results_DMNN.txt

for j in 2 3 4 5
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../../dmnn_torch.py --dataset sepsis24 --n_clusters $j --n_runs 21 --n_epochs 100 --n_z 32 --eta 25 --sub_epochs False --data_ratio -1 >> Results/results_DMNN.txt
done


# echo "DMNN" >> Results/results_DMNN.txt
# echo "CIC 48" >> Results/results_DMNN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../../dmnn_torch.py --dataset cic --n_clusters $j --n_runs 5 --n_epochs 100 --n_z 32 >> Results/results_DMNN.txt
# done
