# echo "KMeans" >> Results/results_KM.txt
# echo "CIC" >> Results/results_KM.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../KMeans.py --dataset cic --n_clusters $j --log_interval 2 --n_epochs 50 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Sepsis" >> Results/results_KM.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../KMeans.py --dataset sepsis --n_clusters $j --log_interval 2 --n_epochs 50 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Kidney" >> Results/results_KM.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../KMeans.py --dataset aki --n_clusters $j --log_interval 2 --n_epochs 50 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Respiratory" >> Results/results_KM.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../KMeans.py --dataset ards --n_clusters $j --log_interval 2 --n_epochs 50 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Wid_Mortality" >> Results/results_KM.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../KMeans.py --dataset wid_mortality --n_clusters $j --log_interval 2 --n_epochs 50 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Diabetes" >> Results/results_KM.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../KMeans.py --dataset diabetes --n_clusters $j --log_interval 2 --n_epochs 50 >> Results/results_KM.txt
# done

echo "KMeans" >> Results/results_KM.txt
echo "CIC_LOS" >> Results/results_KM.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../KMeans.py --dataset cic_los --n_clusters $j --log_interval 2 --n_epochs 50 --n_classes 3  >> Results/results_KM.txt
done