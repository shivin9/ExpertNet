########################################################
########################################################
##################### DeepCAC Exp. #####################
########################################################
########################################################

# echo "DeepCAC expts" >> Results/results_CAC2.txt

# echo "DeepCAC" >> Results/results_CAC.txt
# echo "CIC" >> Results/results_CAC.txt

# for j in 1 2 3 4 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../../deepcac.py --dataset cic --n_clusters $j --n_runs 5 --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
# done


# echo "DeepCAC" >> Results/results_CAC.txt
# echo "CIC_LOS" >> Results/results_CAC.txt

# for j in 1 2 3 4 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../../deepcac.py --dataset cic_los --n_clusters $j --n_runs 5 --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 10 --log_interval 2 --attention False --verbose False --n_epochs 100 --n_classes 3 --n_z 32 >> Results/results_CAC.txt
# done


# echo "DeepCAC" >> Results/results_CAC.txt
# echo "Wid_Mortality" >> Results/results_CAC.txt

# for j in 1 2 3 4 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_CAC.txt
#     python3 ../../deepcac.py --dataset wid_mortality --n_clusters $j --n_runs 5 --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
# done


echo "DeepCAC" >> Results/results_CAC.txt
echo "Sepsis24" >> Results/results_CAC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../../deepcac.py --dataset sepsis24 --n_clusters $j --n_runs 5 --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done


echo "DeepCAC" >> Results/results_CAC.txt
echo "Respiratory24" >> Results/results_CAC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../../deepcac.py --dataset ards24 --n_clusters $j --n_runs 5 --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done


echo "DeepCAC" >> Results/results_CAC.txt
echo "Diabetes" >> Results/results_CAC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../../deepcac.py --dataset diabetes --n_clusters $j --n_runs 5 --alpha 1 --beta 2 --gamma 0 --delta 0 --eta 5 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done