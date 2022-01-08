echo "ExpertNet" >> Results/results_SAE.txt
echo "CIC" >> Results/results_SAE.txt

python3 ../deepcac.py --dataset cic --n_clusters 1 --alpha 1 --beta 0 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/results_SAE.txt

echo "ExpertNet" >> Results/results_SAE.txt
echo "Sepsis" >> Results/results_SAE.txt

python3 ../deepcac.py --dataset sepsis --n_clusters 1 --alpha 1 --beta 0 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/results_SAE.txt

echo "ExpertNet" >> Results/results_SAE.txt
echo "Kidney" >> Results/results_SAE.txt

python3 ../deepcac.py --dataset aki --n_clusters 1 --alpha 1 --beta 0 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/results_SAE.txt

echo "ExpertNet" >> Results/results_SAE.txt
echo "Respiratory" >> Results/results_SAE.txt

python3 ../deepcac.py --dataset ards --n_clusters 1 --alpha 1 --beta 0 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/results_SAE.txt

echo "ExpertNet" >> Results/results_SAE.txt
echo "Wid_Mortality" >> Results/results_SAE.txt

python3 ../deepcac.py --dataset wid_mortality --n_clusters 1 --alpha 1 --beta 0 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/results_SAE.txt

echo "ExpertNet" >> Results/results_SAE.txt
echo "Diabetes" >> Results/results_SAE.txt

python3 ../deepcac.py --dataset diabetes --n_clusters 1 --alpha 1 --beta 0 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/results_SAE.txt

echo "ExpertNet" >> Results/results_SAE.txt
echo "CIC_LOS" >> Results/results_SAE.txt

python3 ../deepcac.py --dataset cic_los --n_clusters 1 --alpha 1 --beta 0 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 --n_classes 3 >> Results/results_SAE.txt
