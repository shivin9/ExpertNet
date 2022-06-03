echo "DMNN" >> Results/results_DMNN.txt
echo "Kidney New" >> Results/results_DMNN.txt

for j in 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../../dmnn_torch.py --dataset aki_new --n_clusters $j --n_runs 5 --n_epochs 100  >> Results/results_DMNN.txt
done


echo "DMNN" >> Results/results_DMNN.txt
echo "Respiratory New" >> Results/results_DMNN.txt

for j in 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../../dmnn_torch.py --dataset ards_new --n_clusters $j --n_runs 5 --n_epochs 100  >> Results/results_DMNN.txt
done


echo "DMNN" >> Results/results_DMNN.txt
echo "CIC New" >> Results/results_DMNN.txt

for j in 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../../dmnn_torch.py --dataset cic_new --n_clusters $j --n_runs 5 --n_epochs 100  >> Results/results_DMNN.txt
done


echo "DMNN" >> Results/results_DMNN.txt
echo "CIC-LoS New" >> Results/results_DMNN.txt

for j in 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../../dmnn_torch.py --dataset cic_los_new --n_clusters $j --n_runs 5 --n_epochs 100 --n_classes 3 >> Results/results_DMNN.txt
done
