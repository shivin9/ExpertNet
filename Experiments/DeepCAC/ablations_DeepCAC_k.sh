echo "DeepCAC Ablations" >> Results/DeepCAC_Ablations_k.txt

python3 ../../deepcac.py --dataset cic --n_clusters 2 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 4 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 5 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 10 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 15 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 20 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 30 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 40 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
python3 ../../deepcac.py --dataset cic --n_clusters 50 --n_classes 2 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_k.txt
