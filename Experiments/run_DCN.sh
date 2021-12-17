echo "DCN" >> Results/results_DCN.txt
echo "CIC" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset cic --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Sepsis" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset sepsis --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Kidney" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset aki --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Respiratory" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset ards --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Wid_Mortality" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Diabetes" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
done