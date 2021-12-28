echo "DMNN" >> Results/results_DMNN.txt
echo "CIC" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn.py --dataset cic --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Sepsis" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn.py --dataset sepsis --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Kidney" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn.py --dataset aki --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Respiratory" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn.py --dataset ards --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Wid_Mortality" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn.py --dataset wid_mortality --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Diabetes" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn.py --dataset diabetes --n_clusters $j >> Results/results_DMNN.txt
done