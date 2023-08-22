###########################
### DEEPCAC Experiments ###
###########################
echo "DeepCAC" >> Results/results_KM.txt

echo "KMeans" >> Results/results_KM.txt
echo "CIC_LOS" >> Results/results_KM.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset cic_los --n_clusters $j --log_interval 2 --n_runs 5 --optimize loss --expt DeepCAC --n_epochs 50 --n_z 32 --n_classes 3  >> Results/results_KM.txt
done


echo "KMeans" >> Results/results_KM.txt
echo "CIC" >> Results/results_KM.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset cic --n_clusters $j --log_interval 2 --n_runs 5 --optimize loss --expt DeepCAC --n_epochs 50 --n_z 32 >> Results/results_KM.txt
done


echo "KMeans" >> Results/results_KM.txt
echo "Respiratory" >> Results/results_KM.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset ards24 --n_clusters $j --log_interval 2 --n_runs 5 --optimize loss --expt DeepCAC --n_epochs 50 --n_z 32 >> Results/results_KM.txt
done


echo "KMeans" >> Results/results_KM.txt
echo "Sepsis24" >> Results/results_KM.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset sepsis24 --n_clusters $j --log_interval 2 --n_runs 5 --optimize loss --expt DeepCAC --n_epochs 50 --n_z 32 >> Results/results_KM.txt
done


echo "KMeans" >> Results/results_KM.txt
echo "Wid_Mortality" >> Results/results_KM.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset wid_mortality --n_clusters $j --log_interval 2 --n_runs 5 --optimize loss --expt DeepCAC --n_epochs 50 --n_z 32 >> Results/results_KM.txt
done


echo "KMeans" >> Results/results_KM.txt
echo "Diabetes" >> Results/results_KM.txt

for j in 5 8 10 15 20 30 50
do
    echo "k = $(($j))" >> Results/results_KM.txt
    python3 ../../KMeans.py --dataset diabetes --n_clusters $j --log_interval 2 --n_runs 5 --optimize loss --expt DeepCAC --n_epochs 50 --n_z 32 >> Results/results_KM.txt
done
