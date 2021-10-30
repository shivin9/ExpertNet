echo "DCN" >> results_IDEC.txt
echo "CIC" >> results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> results_IDEC.txt
    python3 IDEC.py --dataset cic --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> results_IDEC.txt
done

echo "IDEC" >> results_IDEC.txt
echo "Sepsis" >> results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> results_IDEC.txt
    python3 IDEC.py --dataset sepsis --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> results_IDEC.txt
done

echo "IDEC" >> results_IDEC.txt
echo "Kidney" >> results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> results_IDEC.txt
    python3 IDEC.py --dataset aki --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> results_IDEC.txt
done

echo "IDEC" >> results_IDEC.txt
echo "Respiratory" >> results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> results_IDEC.txt
    python3 IDEC.py --dataset ards --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> results_IDEC.txt
done

echo "IDEC" >> results_IDEC.txt
echo "Wid_Mortality" >> results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> results_IDEC.txt
    python3 IDEC.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> results_IDEC.txt
done

echo "IDEC" >> results_IDEC.txt
echo "Diabetes" >> results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> results_IDEC.txt
    python3 IDEC.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> results_IDEC.txt
done