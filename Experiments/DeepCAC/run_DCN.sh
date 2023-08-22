######################################################################
# DEEPCAC Baselines
######################################################################
echo "DeepCAC" >> Results/results_DCN.txt

echo "DCN" >> Results/results_DCN.txt
echo "CIC" >> Results/results_DCN.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset cic --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --optimize loss --expt DeepCAC >> Results/results_DCN.txt
done


echo "DCN" >> Results/results_DCN.txt
echo "CIC_LOS" >> Results/results_DCN.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset cic_los --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --optimize loss --expt DeepCAC --n_classes 3 >> Results/results_DCN.txt
done


echo "DCN" >> Results/results_DCN.txt
echo "Respiratory" >> Results/results_DCN.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset ards24 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --optimize loss --expt DeepCAC >> Results/results_DCN.txt
done


echo "DeepCAC DCN" >> Results/results_DCN.txt
echo "Sepsis24" >> Results/results_DCN.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset sepsis24 --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --optimize loss --expt DeepCAC >> Results/results_DCN.txt
done


echo "DCN" >> Results/results_DCN.txt
echo "Wid_Mortality" >> Results/results_DCN.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset wid_mortality --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --optimize loss --expt DeepCAC >> Results/results_DCN.txt
done


echo "DCN" >> Results/results_DCN.txt
echo "Diabetes" >> Results/results_DCN.txt

for j in 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_DCN.txt
    python3 ../../DCN.py --dataset diabetes --n_clusters $j --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_z 32 --n_runs 5 --optimize loss --expt DeepCAC >> Results/results_DCN.txt
done
