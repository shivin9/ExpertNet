echo "DeepCAC Ablations" >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.00 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.001 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.005 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.01 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.02 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.05 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.1 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 0.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 1.0 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 1.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 10 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 20 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 50 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
python3 ../../deepcac.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --eta 100 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt

# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.00 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.001 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.005 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.01 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.02 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.05 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.1 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 0.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 1.0 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 1.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 10 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 20 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 50 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset ards24 --n_clusters 3 --alpha 1 --beta 2 --eta 100 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt


# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.00 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.001 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.005 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.01 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.02 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.05 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.1 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 0.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 1.0 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 1.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 10 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset adult --n_clusters 3 --alpha 1 --beta 2 --eta 100 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt

# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.00 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.001 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.005 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.01 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.02 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.05 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.1 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.2 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 1.0 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 1.5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 2 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 5 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 10 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 100 --log_interval 2 --n_epochs 100 --n_runs 3 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt

# python3 ../../deepcac.py --dataset cic_los --n_clusters 2 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 3 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 4 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 5 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 10 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 15 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 20 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 30 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 40 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
# python3 ../../deepcac.py --dataset cic_los --n_clusters 50 --n_classes 3 --alpha 1 --beta 20 --eta 0.5 --log_interval 2 --n_epochs 100 --n_runs 1 --n_z 32 >> Results/DeepCAC_Ablations_eta.txt
