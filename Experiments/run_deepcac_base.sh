echo "CAC" >> Results/results_DeepCAC_base.txt
echo "CIC" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset cic --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DeepCAC_base.txt
done

echo "CAC" >> Results/results_DeepCAC_base.txt
echo "Sepsis" >> Results/results_DeepCAC_base.txt

# for j in 1
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
#     python3 ../deepcac.py --dataset sepsis --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DeepCAC_base.txt
# done

# echo "CAC" >> Results/results_DeepCAC_base.txt
# echo "Kidney" >> Results/results_DeepCAC_base.txt

# for j in 1
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
#     python3 ../deepcac.py --dataset aki --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DeepCAC_base.txt
# done

# echo "CAC" >> Results/results_DeepCAC_base.txt
# echo "Respiratory" >> Results/results_DeepCAC_base.txt

# for j in 1
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
#     python3 ../deepcac.py --dataset ards --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DeepCAC_base.txt
# done

# echo "CAC" >> Results/results_DeepCAC_base.txt
# echo "Wid_Mortality" >> Results/results_DeepCAC_base.txt

# for j in 1
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
#     python3 ../deepcac.py --dataset wid_mortality --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DeepCAC_base.txt
# done

echo "CAC" >> Results/results_DeepCAC_base.txt
echo "Diabetes" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset diabetes --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --attention False --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DeepCAC_base.txt
done

echo "CAC" >> Results/results_DeepCAC_base.txt
echo "CIC_LOS" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset cic_los --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --log_interval 2 --attention False --n_epochs 50 --n_classes 3 --n_z 32 >> Results/results_DeepCAC_base.txt
done

echo "CAC" >> Results/results_DeepCAC_base.txt
echo "Adult" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset adult --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_DeepCAC_base.txt
done

echo "CAC" >> Results/results_DeepCAC_base.txt
echo "Creditcard" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset creditcard --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_DeepCAC_base.txt
done

echo "CAC" >> Results/results_DeepCAC_base.txt
echo "Magic" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset magic --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_DeepCAC_base.txt
done


echo "CAC" >> Results/results_DeepCAC_base.txt
echo "Titanic" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset titanic --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_DeepCAC_base.txt
done


echo "CAC" >> Results/results_DeepCAC_base.txt
echo "Heart" >> Results/results_DeepCAC_base.txt

for j in 1
do
    echo "k = $(($j))" >> Results/results_DeepCAC_base.txt
    python3 ../deepcac.py --dataset heart --n_clusters $j --alpha 0 --beta 0 --gamma 0 --delta 0 --eta 0 --log_interval 2 --attention False --n_epochs 50 --n_classes 2 --n_z 32 >> Results/results_DeepCAC_base.txt
done