echo "DeepCAC Ablations" >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.00 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.001 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.01 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.05 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.1 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.2 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.5 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.0 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.5 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 5 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 10 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 50 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 100 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_beta.txt