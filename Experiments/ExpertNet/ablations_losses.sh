echo "All: --alpha 1 --beta 2 --gamma 10 --delta 0.1" >> Results/ExpertNet_Ablations_losses.txt

python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 1 --beta 2 --gamma 10 --delta 0.1 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_losses.txt

echo "Delta = 0" >> Results/ExpertNet_Ablations_losses.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 1 --beta 2 --gamma 10 --delta 0 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_losses.txt

echo "Delta,Beta = 0" >> Results/ExpertNet_Ablations_losses.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 1 --beta 0 --gamma 10 --delta 0 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_losses.txt

echo "Delta,Alpha = 0" >> Results/ExpertNet_Ablations_losses.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 0 --beta 2 --gamma 10 --delta 0 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_losses.txt

echo "Delta,Beta,Alpha = 0" >> Results/ExpertNet_Ablations_losses.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 0 --beta 0 --gamma 10 --delta 0 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_losses.txt

echo "Delta,Beta,Alpha,Gamma = 0" >> Results/ExpertNet_Ablations_losses.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 0 --beta 0 --gamma 0 --delta 0 --log_interval 2 --n_runs 3 --n_z 32 --n_epochs 50 >> Results/ExpertNet_Ablations_losses.txt
