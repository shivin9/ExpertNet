# echo "CAC" >> Results/results_CAC.txt
# echo "CIC" >> Results/results_CAC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset cic --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_CAC.txt
# done

# echo "CAC" >> Results/results_CAC.txt
# echo "Sepsis" >> Results/results_CAC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset sepsis --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_CAC.txt
# done

# echo "CAC" >> Results/results_CAC.txt
# echo "Kidney" >> Results/results_CAC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset aki --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_CAC.txt
# done

# echo "CAC" >> Results/results_CAC.txt
# echo "Respiratory" >> Results/results_CAC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset ards --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_CAC.txt
# done

# echo "CAC" >> Results/results_CAC.txt
# echo "Wid_Mortality" >> Results/results_CAC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_CAC.txt
# done

echo "CAC" >> Results/results_CAC.txt
echo "Diabetes" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset diabetes --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 50 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_CAC.txt
done

# echo "CAC" >> Results/results_CAC.txt
# echo "CIC_LOS" >> Results/results_CAC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset cic_los --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --log_interval 2 --attention False --n_epochs 50 --n_classes 3 --n_z 32 >> Results/results_CAC.txt
# done

# echo "CAC" >> Results/results_CAC.txt
# echo "Adult" >> Results/results_CAC.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset adult --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_CAC.txt
# done

# echo "CAC" >> Results/results_CAC.txt
# echo "Creditcard" >> Results/results_CAC.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset creditcard --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_CAC.txt
# done

# echo "CAC" >> Results/results_CAC.txt
# echo "Magic" >> Results/results_CAC.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset magic --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_CAC.txt
# done


# echo "CAC" >> Results/results_CAC.txt
# echo "Titanic" >> Results/results_CAC.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset titanic --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_CAC.txt
# done


# echo "CAC" >> Results/results_CAC.txt
# echo "Heart" >> Results/results_CAC.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../deepcac.py --dataset heart --n_clusters $j --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_CAC.txt
# done