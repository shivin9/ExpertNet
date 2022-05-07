# echo "DCN" >> Results/results_DCN.txt
# echo "Sepsis" >> Results/results_DCN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../DCN.py --dataset sepsis --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Kidney" >> Results/results_DCN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../DCN.py --dataset aki --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Respiratory" >> Results/results_DCN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../DCN.py --dataset ards --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Wid_Mortality" >> Results/results_DCN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../DCN.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Kidney New" >> Results/results_DCN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../DCN.py --dataset aki_new --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Respiratory New" >> Results/results_DCN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../DCN.py --dataset ards_new --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "CIC New" >> Results/results_DCN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../DCN.py --dataset cic_new --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 >> Results/results_DCN.txt
# done


######################################################################
# DEEPCAC Baselines
######################################################################
echo "DCN" >> Results/results_DCN.txt
echo "CIC" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset cic --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Diabetes" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset diabetes --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "CIC_LOS" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset cic_los --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_classes 3 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Titanic" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset titanic --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Heart" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset heart --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Adult" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset adult --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
done

echo "DCN" >> Results/results_DCN.txt
echo "Creditcard" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset creditcard --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
done

echo "DeepCAC DCN" >> Results/results_DCN.txt
echo "Sepsis" >> Results/results_DCN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../DCN.py --dataset sepsis --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
done
