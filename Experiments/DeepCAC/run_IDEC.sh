########################################################################################
# DeepCAC Baselines
########################################################################################
echo "DeepCAC" >> Results/results_IDEC.txt


echo "IDEC" >> Results/results_IDEC.txt
echo "CIC" >> Results/results_IDEC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset cic --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 5 --n_z 32 --optimize loss --expt DeepCAC >> Results/results_IDEC.txt
done


echo "IDEC" >> Results/results_IDEC.txt
echo "CIC_LOS" >> Results/results_IDEC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset cic_los --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 5 --n_z 32 --n_classes 3 --optimize loss --expt DeepCAC >> Results/results_IDEC.txt
done


echo "IDEC" >> Results/results_IDEC.txt
echo "Respiratory24" >> Results/results_IDEC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset ards24 --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 5 --n_z 32 --optimize loss --expt DeepCAC >> Results/results_IDEC.txt
done


echo "DeepCAC IDEC" >> Results/results_IDEC.txt
echo "Sepsis24" >> Results/results_IDEC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset sepsis24 --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 5 --n_z 32 --optimize loss --expt DeepCAC >> Results/results_IDEC.txt
done


echo "IDEC" >> Results/results_IDEC.txt
echo "Wid_Mortality" >> Results/results_IDEC.txt

for j in 1 2 3 4 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_IDEC.txt
    python3 ../../IDEC.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 5 --n_z 32 --optimize loss --expt DeepCAC >> Results/results_IDEC.txt
done

# echo "IDEC" >> Results/results_IDEC.txt
# echo "Diabetes" >> Results/results_IDEC.txt

# for j in 1 2 3 4 5 8 10 15 20 30 50
# do
#     echo "k = $(($j))" >> Results/results_IDEC.txt
#     python3 ../../IDEC.py --dataset diabetes --n_clusters $j --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 100 --n_runs 5 --n_z 32 --optimize loss --expt DeepCAC  >> Results/results_IDEC.txt
# done