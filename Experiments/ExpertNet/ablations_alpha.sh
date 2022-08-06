echo "alpha" >> Results/ExpertNet_Ablations_alpha.txt

python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 0 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 0.01 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 0.1  --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 0.2  --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 0.5  --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 2 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 5 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 10 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 20 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 50 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 100  --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_alpha.txt