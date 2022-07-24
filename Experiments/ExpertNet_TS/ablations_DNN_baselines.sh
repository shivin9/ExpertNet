echo "KMeans Ablations" >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.00 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.001 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.005 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.01 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.02 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.05 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.1 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.0 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 10 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt
python3 ../KMeans.py --dataset cic --n_clusters 3 --alpha 1 --beta 100 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/KM_Ablations.txt

echo "DCN Ablations" >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.00 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.001 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.005 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.01 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.02 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.05 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.1 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.0 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 10 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt
python3 ../DCN.py --dataset cic --n_clusters 3 --alpha 1 --beta 100 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/DCN_Ablations.txt

echo "IDEC Ablations" >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.00 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.001 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.005 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.01 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.02 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.05 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.1 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 0.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.0 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 1.5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 2 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 5 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 10 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
python3 ../IDEC.py --dataset cic --n_clusters 3 --alpha 1 --beta 100 --log_interval 2 --n_epochs 50 --n_runs 3 --n_z 32 >> Results/IDEC_Ablations.txt
