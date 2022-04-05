echo "DCN" >> Results/results_IDEC.txt
echo "CIC" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset cic --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "Sepsis" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset sepsis --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "Kidney" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset aki --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "Respiratory" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset ards --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "Wid_Mortality" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "Diabetes" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "CIC_LOS" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset cic_los --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 --n_classes 3 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "Kidney New" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset aki_new --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "Respiratory New" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset ards_new --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "CIC New" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../IDEC.py --dataset cic_new --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 >> Results/results_IDEC.txt
done
