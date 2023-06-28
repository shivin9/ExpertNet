echo "delta" >> Results/ExpertNet_Ablations_delta.txt

python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.01 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.2 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.5 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 1 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 2 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 5 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 10 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 20 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 50 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 100 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt

python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.01 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.2 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.5 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 1 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 2 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 5 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 10 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 20 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 50 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 100 --eta 0 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 >> Results/ExpertNet_Ablations_delta.txt