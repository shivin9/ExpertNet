echo "gamma" >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.001 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.01 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.1 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.2 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.5 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 1 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 1.5 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 2 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 3 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 10 --delta 0.1 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt


# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.01 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 0.5 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 1 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 1.5 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 2 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt
# python3 ../../expertnet.py --dataset cic_new --n_clusters 3 --alpha 1 --beta 10 --gamma 10 --delta 0 --eta 0.0 --log_interval 2 --n_epochs 50 >> Results/ExpertNet_Ablations_gamma.txt