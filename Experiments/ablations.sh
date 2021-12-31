echo "ExpertNet Ablations" >> Results/ExpertNet_Ablations.txt
echo "alpha" >> Results/ExpertNet_Ablations_alpha.txt

python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0.001 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0.005 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0.01 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0.05 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0.1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0.2 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 0.5 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 2 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 5 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 10 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt