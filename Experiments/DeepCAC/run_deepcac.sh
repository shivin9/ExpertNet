echo "DeepCAC" >> Results/results_CAC.txt
echo "CIC" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset cic --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 1 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done


echo "DeepCAC" >> Results/results_CAC.txt
echo "Sepsis" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset sepsis --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 1 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done

echo "DeepCAC" >> Results/results_CAC.txt
echo "Kidney" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset aki --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 1 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done

echo "DeepCAC" >> Results/results_CAC.txt
echo "Respiratory" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset ards --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 1 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done

echo "DeepCAC" >> Results/results_CAC.txt
echo "Diabetes" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset diabetes --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 10 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done

echo "DeepCAC" >> Results/results_CAC.txt
echo "CIC_LOS" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset cic_los --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 1 --log_interval 2 --attention False --verbose False --n_epochs 100 --n_classes 3 --n_z 32 >> Results/results_CAC.txt
done

echo "DeepCAC" >> Results/results_CAC.txt
echo "Heart" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset heart --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 1 --log_interval 2 --attention False --verbose False --n_epochs 100 --n_classes 2 --n_z 32 >> Results/results_CAC.txt
done

echo "DeepCAC" >> Results/results_CAC.txt
echo "Wid_Mortality" >> Results/results_CAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset wid_mortality --n_clusters $j --n_runs 3 --alpha 1 --beta 20 --gamma 0 --delta 0 --eta 1 --attention False --log_interval 2 --verbose False --n_epochs 100 --n_z 32 >> Results/results_CAC.txt
done