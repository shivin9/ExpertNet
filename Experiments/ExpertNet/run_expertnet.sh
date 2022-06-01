########################################################
########################################################
################### ExpertNET Exp. #####################
########################################################
########################################################

echo "ExpertNet Experiments" >> Results/results_EN.txt

echo "ExpertNet" >> Results/results_EN.txt
echo "Kidney New" >> Results/results_EN.txt

for j in 1 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_EN.txt
    python3 ../../expertnet.py --dataset aki_new --n_clusters $j --alpha 1 --beta 2 --gamma 2.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 100 >> Results/results_EN.txt
done

echo "ExpertNet" >> Results/results_EN.txt
echo "Respiratory New" >> Results/results_EN.txt

for j in 1 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_EN.txt
    python3 ../../expertnet.py --dataset ards_new --n_clusters $j --alpha 1 --beta 2 --gamma 2.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 100 >> Results/results_EN.txt
done

echo "ExpertNet" >> Results/results_EN.txt
echo "CIC New" >> Results/results_EN.txt

for j in 1 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_EN.txt
    python3 ../../expertnet.py --dataset cic_new --n_clusters $j --alpha 1 --beta 2 --gamma 2.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 100 >> Results/results_EN.txt
done

echo "ExpertNet" >> Results/results_EN.txt
echo "CIC LoS New" >> Results/results_EN.txt

for j in 1 2 3 4 5 6 7 8 10
do
    echo "k = $(($j))" >> Results/results_EN.txt
    python3 ../../expertnet.py --dataset cic_los_new --n_clusters $j --alpha 1 --beta 2 --gamma 2.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 100  --n_classes 3 >> Results/results_EN.txt
done
