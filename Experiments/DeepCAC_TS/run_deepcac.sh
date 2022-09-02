########################################################
########################################################
################### ExpertNET Exp. #####################
########################################################
########################################################

# echo "ExpertNet Experiments" >> Results/results_DeepCAC.txt

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "Kidney New" >> Results/results_DeepCAC.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet.py --dataset aki_new --n_clusters $j --alpha 1 --beta 2 --alpha 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_DeepCAC.txt
# done

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "Respiratory New" >> Results/results_DeepCAC.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet.py --dataset ards_new --n_clusters $j --alpha 1 --beta 2 --alpha 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_DeepCAC.txt
# done

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "CIC New" >> Results/results_DeepCAC.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet.py --dataset cic_new --n_clusters $j --alpha 1 --beta 2 --alpha 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_DeepCAC.txt
# done

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "CIC LoS New" >> Results/results_DeepCAC.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet.py --dataset cic_los_new --n_clusters $j --alpha 1 --beta 2 --alpha 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32   --n_classes 3 >> Results/results_DeepCAC.txt
# done

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "IHM" >> Results/results_DeepCAC.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet.py --dataset ihm --n_clusters $j --alpha 1 --beta 2 --alpha 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32   --n_classes 3 >> Results/results_DeepCAC.txt
# done

########################################################
########################################################
################ ExpertNET New Exp. ####################
########################################################
########################################################


echo "ExpertNet Student Experiments" >> Results/results_DeepCAC.txt

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "Kidney 48" >> Results/results_DeepCAC.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet.py --dataset aki48 --n_clusters $j --alpha 1 --beta 2 --alpha 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_DeepCAC.txt
# done

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "Respiratory TS" >> Results/results_DeepCAC.txt

# for j in 2 3 4 5
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet_gru.py --dataset ards_ts --n_clusters $j --beta 10 --alpha 5 --delta 0.1 --n_epochs 100 --n_z 45 --n_features 89 >> Results/results_DeepCAC.txt
# done


echo "ExpertNet" >> Results/results_DeepCAC.txt
echo "Sepsis TS" >> Results/results_DeepCAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_DeepCAC.txt
    python3 ../../deepcac_gru.py --dataset sepsis_ts --n_clusters $j --beta 20 --alpha 5 --n_epochs 10 --n_z 32 --n_features 44 --n_runs 3 >> Results/results_DeepCAC.txt
done


echo "ExpertNet" >> Results/results_DeepCAC.txt
echo "ARDS TS" >> Results/results_DeepCAC.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_DeepCAC.txt
    python3 ../../deepcac_gru.py --dataset ards_ts --n_clusters $j --beta 20 --alpha 5 --n_epochs 10 --n_z 32  --n_features 89 --n_runs 3 >> Results/results_DeepCAC.txt
done

# echo "ExpertNet" >> Results/results_DeepCAC.txt
# echo "CIC 48" >> Results/results_DeepCAC.txt

# for j in 1 2 3 4 5 6 7 8 10
# do
#     echo "k = $(($j))" >> Results/results_DeepCAC.txt
#     python3 ../../expertnet.py --dataset cic --n_clusters $j --alpha 1 --beta 2 --alpha 10 --delta 0.1 --log_interval 2 --n_epochs 100 --n_z 32  >> Results/results_DeepCAC.txt
# done