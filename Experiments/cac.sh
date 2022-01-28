echo "CAC" >> Results/results_CAC.txt
echo "CIC" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset cic --n_clusters $j --alpha 1 --beta 0 --gamma 0 --delta 0 --eta 0.04 --attention False --log_interval 2 --n_epochs 50 >> Results/results_CAC.txt
done

echo "CAC" >> Results/results_CAC.txt
echo "Sepsis" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset sepsis --n_clusters $j --alpha 1 --beta 0 --gamma 0 --delta 0 --eta 0.04 --attention False --log_interval 2 --n_epochs 50 >> Results/results_CAC.txt
done

echo "CAC" >> Results/results_CAC.txt
echo "Kidney" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset aki --n_clusters $j --alpha 1 --beta 0 --gamma 0 --delta 0 --eta 0.04 --attention False --log_interval 2 --n_epochs 50 >> Results/results_CAC.txt
done

echo "CAC" >> Results/results_CAC.txt
echo "Respiratory" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset ards --n_clusters $j --alpha 1 --beta 0 --gamma 0 --delta 0 --eta 0.04 --attention False --log_interval 2 --n_epochs 50 >> Results/results_CAC.txt
done

echo "CAC" >> Results/results_CAC.txt
echo "Wid_Mortality" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0 --gamma 0 --delta 0 --eta 0.04 --attention False --log_interval 2 --n_epochs 50 >> Results/results_CAC.txt
done

echo "CAC" >> Results/results_CAC.txt
echo "Diabetes" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0 --gamma 0 --delta 0 --eta 0.04 --attention False --log_interval 2 --n_epochs 50 >> Results/results_CAC.txt
done

echo "CAC" >> Results/results_CAC.txt
echo "CIC_LOS" >> Results/results_CAC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_CAC.txt
    python3 ../deepcac.py --dataset cic_los --n_clusters $j --alpha 1 --beta 0 --gamma 0 --delta 0 --eta 0.04 --log_interval 2 --attention False --n_epochs 50 --n_classes 3 >> Results/results_CAC.txt
done