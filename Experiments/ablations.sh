echo "ExpertNet Ablations" >> Results/ExpertNet_Ablations.txt

python3 ../deepcac.py --dataset cic --n_clusters 2 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt
python3 ../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt
python3 ../deepcac.py --dataset cic --n_clusters 4 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt
python3 ../deepcac.py --dataset cic --n_clusters 5 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt


python3 ../deepcac.py --dataset sepsis --n_clusters 2 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt
python3 ../deepcac.py --dataset sepsis --n_clusters 3 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt
python3 ../deepcac.py --dataset sepsis --n_clusters 4 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt
python3 ../deepcac.py --dataset sepsis --n_clusters 5 --alpha 1 --beta 0.5 --gamma 1.5 --delta 1 --eta 0.0 --log_interval 2 --n_epochs 50 --attention False >> Results/ExpertNet_Ablations.txt