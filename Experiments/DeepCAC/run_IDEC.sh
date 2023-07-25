########################################################################################
# DeepCAC Baselines
########################################################################################
echo "DeepCAC" >> Results/results_IDEC.txt

echo "IDEC" >> Results/results_IDEC.txt
echo "Diabetes" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC  >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "CIC" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset cic --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
done

echo "IDEC" >> Results/results_IDEC.txt
echo "CIC_LOS" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset cic_los --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --n_classes 3 --expt DeepCAC >> Results/results_IDEC.txt
done

# echo "IDEC" >> Results/results_IDEC.txt
# echo "Titanic" >> Results/results_IDEC.txt

# for j in 1 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset titanic --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 >> Results/results_IDEC.txt
# done

# echo "IDEC" >> Results/results_IDEC.txt
# echo "Heart" >> Results/results_IDEC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset heart --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
# done

# echo "IDEC" >> Results/results_IDEC.txt
# echo "Creditcard" >> Results/results_IDEC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset creditcard --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 >> Results/results_IDEC.txt
# done


# echo "IDEC" >> Results/results_IDEC.txt
# echo "Adult" >> Results/results_IDEC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset adult --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC  >> Results/results_IDEC.txt
# done

# echo "DeepCAC IDEC" >> Results/results_IDEC.txt
# echo "Sepsis" >> Results/results_IDEC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset sepsis --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
# done

# echo "IDEC" >> Results/results_IDEC.txt
# echo "Kidney" >> Results/results_IDEC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset aki --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
# done

# echo "IDEC" >> Results/results_IDEC.txt
# echo "Respiratory" >> Results/results_IDEC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset ards --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
# done


echo "IDEC" >> Results/results_IDEC.txt
echo "Respiratory24" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset ards24 --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
done

echo "DeepCAC IDEC" >> Results/results_IDEC.txt
echo "Sepsis24" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset sepsis24 --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
done

# echo "IDEC" >> Results/results_IDEC.txt
# echo "Kidney24" >> Results/results_IDEC.txt

# for j in 1 2 3 4
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset aki24 --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
# done


echo "IDEC" >> Results/results_IDEC.txt
echo "Wid_Mortality" >> Results/results_IDEC.txt

for j in 1 2 3 4
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 11 --n_z 32 --expt DeepCAC >> Results/results_IDEC.txt
done