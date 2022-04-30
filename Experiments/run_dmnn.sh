echo "DMNN" >> Results/results_DMNN.txt
echo "CIC" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn_torch.py --dataset cic --n_clusters $j >> Results/results_DMNN.txt
done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Sepsis" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset sepsis --n_clusters $j >> Results/results_DMNN.txt
# done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Kidney" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset aki --n_clusters $j >> Results/results_DMNN.txt
# done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Respiratory" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset ards --n_clusters $j >> Results/results_DMNN.txt
# done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Wid_Mortality" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset wid_mortality --n_clusters $j >> Results/results_DMNN.txt
# done

echo "DMNN" >> Results/results_DMNN.txt
echo "Diabetes" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn_torch.py --dataset diabetes --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Titanic" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn_torch.py --dataset titanic --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Adult" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn_torch.py --dataset adult --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Creditcard" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn_torch.py --dataset creditcard --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "Heart" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn_torch.py --dataset heart --n_clusters $j >> Results/results_DMNN.txt
done

echo "DMNN" >> Results/results_DMNN.txt
echo "CIC_LOS" >> Results/results_DMNN.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_DMNN.txt
    python3 ../dmnn_torch.py --dataset cic_los --n_clusters $j  --n_classes 3 >> Results/results_DMNN.txt
done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Magic" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset magic --n_clusters $j >> Results/results_DMNN.txt
# done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Kidney New" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset aki_new --n_clusters $j >> Results/results_DMNN.txt
# done

# echo "DMNN" >> Results/results_DMNN.txt
# echo "Respiratory New" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset ards_new --n_clusters $j >> Results/results_DMNN.txt
# done


# echo "DMNN" >> Results/results_DMNN.txt
# echo "CIC New" >> Results/results_DMNN.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DMNN.txt
#     python3 ../dmnn_torch.py --dataset cic_new --n_clusters $j >> Results/results_DMNN.txt
# done