# echo "DCN" >> Results/results_DCN.txt
# echo "Sepsis" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset sepsis --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Kidney" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset aki --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Respiratory" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset ards --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Wid_Mortality" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Kidney New" >> Results/results_DCN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset aki_new --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Respiratory New" >> Results/results_DCN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset ards_new --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "CIC New" >> Results/results_DCN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset cic_new --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "CIC_LoS New" >> Results/results_DCN.txt

# for j in 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset cic_los_new --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_classes 3 >> Results/results_DCN.txt
# done


echo "DCN Experiments" >> Results/results_DCN.txt

# echo "DCN" >> Results/results_DCN.txt
# echo "Kidney 48" >> Results/results_DCN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset aki48 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 100 --n_z 32 --n_runs 21 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Respiratory 24" >> Results/results_DCN.txt

# for j in 2 3 4 5
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset ards24 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 100 --n_z 32 --n_runs 21 >> Results/results_DCN.txt
# done


echo "DCN" >> Results/results_DCN.txt
echo "Sepsis 24" >> Results/results_DCN.txt

for j in 2 3 4 5
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset sepsis24 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 100 --n_z 32 --n_runs 21  >> Results/results_DCN.txt
done


# echo "DCN" >> Results/results_DCN.txt
# echo "CIC 48" >> Results/results_DCN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset cic --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 100 --n_z 32 >> Results/results_DCN.txt
# done
