# echo "KMeans" >> Results/results_KM.txt
# echo "CIC" >> Results/results_KM.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset cic --n_clusters $j --log_interval 2 --n_runs 3 --n_epochs 50 --n_z 32 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Sepsis" >> Results/results_KM.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset sepsis --n_clusters $j --log_interval 2 --n_runs 3 --n_epochs 50 --n_z 32 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Kidney" >> Results/results_KM.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset aki --n_clusters $j --log_interval 2 --n_runs 3 --n_epochs 50 --n_z 32 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Respiratory" >> Results/results_KM.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset ards --n_clusters $j --log_interval 2 --n_runs 3 --n_epochs 50 --n_z 32 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Wid_Mortality" >> Results/results_KM.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset wid_mortality --n_clusters $j --log_interval 2 --n_runs 3 --n_epochs 50 --n_z 32 >> Results/results_KM.txt
# done

# echo "KMeans" >> Results/results_KM.txt
# echo "Diabetes" >> Results/results_KM.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset diabetes --n_clusters $j --log_interval 2 --n_runs 3 --n_epochs 50 --n_z 32 >> Results/results_KM.txt
# done


echo "KMeans Experiments" >> Results/results_KM.txt

# echo "KMeans" >> Results/results_KM.txt
# echo "Kidney 48" >> Results/results_KM.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset aki48 --n_clusters $j --n_runs 3 --log_interval 2 --n_epochs 100 --n_z 32 >> Results/results_KM.txt
# done

echo "KMeans" >> Results/results_KM.txt
echo "Respiratory 24" >> Results/results_KM.txt

# for j in 2 3 4 5
for j in 3 4 5
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset ards24 --n_clusters $j --n_runs 3 --log_interval 2 --n_epochs 100 --n_z 32 --n_runs 21 >> Results/results_KM.txt
done


echo "KMeans" >> Results/results_KM.txt
echo "Sepsis 24" >> Results/results_KM.txt

# for j in 2 3 4 5
for j in 3 4 5
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset sepsis24 --n_clusters $j --n_runs 3 --log_interval 2 --n_epochs 100 --n_z 32 --n_runs 21 >> Results/results_KM.txt
done


# echo "KMeans" >> Results/results_KM.txt
# echo "CIC 48" >> Results/results_KM.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_KM.txt
#     python3 ../../KMeans.py --dataset cic --n_clusters $j --n_runs 3 --log_interval 2 --n_epochs 100 --n_z 32 >> Results/results_KM.txt
# done
