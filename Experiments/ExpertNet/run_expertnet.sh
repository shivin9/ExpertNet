########################################################
########################################################
################### ExpertNET Exp. #####################
########################################################
########################################################

# echo "ExpertNet Experiments" >> Results/results_EN.txt

# echo "ExpertNet" >> Results/results_EN.txt
# echo "Kidney New" >> Results/results_EN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_EN.txt
#     python3 ../../expertnet.py --dataset aki_new --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_EN.txt
# done

# echo "ExpertNet" >> Results/results_EN.txt
# echo "Respiratory New" >> Results/results_EN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_EN.txt
#     python3 ../../expertnet.py --dataset ards_new --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_EN.txt
# done

# echo "ExpertNet" >> Results/results_EN.txt
# echo "CIC New" >> Results/results_EN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_EN.txt
#     python3 ../../expertnet.py --dataset cic_new --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_EN.txt
# done

# echo "ExpertNet" >> Results/results_EN.txt
# echo "CIC LoS New" >> Results/results_EN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_EN.txt
#     python3 ../../expertnet.py --dataset cic_los_new --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32   --n_classes 3 >> Results/results_EN.txt
# done

# echo "ExpertNet" >> Results/results_EN.txt
# echo "IHM" >> Results/results_EN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_EN.txt
#     python3 ../../expertnet.py --dataset ihm --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32   --n_classes 3 >> Results/results_EN.txt
# done

########################################################
########################################################
################ ExpertNET New Exp. ####################
########################################################
########################################################


echo "ExpertNet Student Experiments" >> Results/results_EN.txt

# echo "ExpertNet" >> Results/results_EN.txt
# echo "Kidney 48" >> Results/results_EN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_EN.txt
#     python3 ../../expertnet.py --dataset aki48 --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_EN.txt
# done

echo "ExpertNet" >> Results/results_EN.txt
echo "Respiratory 24" >> Results/results_EN.txt

for j in 2 3 4 5
do
    echo "k = $(($j))" >> Results/results_EN.txt
    python3 ../../expertnet.py --dataset ards24 --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32 --n_runs 21  >> Results/results_EN.txt
done


echo "ExpertNet" >> Results/results_EN.txt
echo "Sepsis 24" >> Results/results_EN.txt

for j in 2 3 4 5
do
    echo "k = $(($j))" >> Results/results_EN.txt
    python3 ../../expertnet.py --dataset sepsis24 --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  --n_runs 21 >> Results/results_EN.txt
done


# echo "ExpertNet" >> Results/results_EN.txt
# echo "CIC 48" >> Results/results_EN.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_EN.txt
#     python3 ../../expertnet.py --dataset cic --n_clusters $j --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_EN.txt
# done
