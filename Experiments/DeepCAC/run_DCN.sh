######################################################################
# DEEPCAC Baselines
######################################################################

# echo "DCN" >> Results/results_DCN.txt
# echo "CIC" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset cic --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Diabetes" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset diabetes --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "CIC_LOS" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset cic_los --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC --n_classes 3 >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Heart" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset heart --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

# echo "DeepCAC DCN" >> Results/results_DCN.txt
# echo "Sepsis" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset sepsis --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Kidney" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset aki --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Respiratory" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset ards --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

# echo "DCN" >> Results/results_DCN.txt
# echo "Wid_Mortality" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

echo "DeepCAC DCN" >> Results/results_DCN.txt
echo "Sepsis24" >> Results/results_DCN.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset sepsis24 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
done

# echo "DCN" >> Results/results_DCN.txt
# echo "Kidney" >> Results/results_DCN.txt

# for j in 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_DCN.txt
#     python3 ../../DCN.py --dataset aki24 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
# done

echo "DCN" >> Results/results_DCN.txt
echo "Respiratory" >> Results/results_DCN.txt

for j in 2 3 4
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset ards24 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --expt DeepCAC >> Results/results_DCN.txt
done
