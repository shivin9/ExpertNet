echo "ExpertNet Ablations" >> Results/ExpertNet_Ablations.txt

echo "Attention Train, Attention Test: F,T" >> Results/ExpertNet_Ablations.txt

python3 ../../expertnet.py --dataset ards --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 6 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 7 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 8 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 9 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 10 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt

echo "Attention Train, Attention Test: F,F" >> Results/ExpertNet_Ablations.txt

python3 ../../expertnet.py --dataset ards --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt

echo "Attention Train, Attention Test: T,F" >> Results/ExpertNet_Ablations.txt

python3 ../../expertnet.py --dataset ards --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt

echo "Attention Train, Attention Test: T,T" >> Results/ExpertNet_Ablations.txt

python3 ../../expertnet.py --dataset ards --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset ards --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 6 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 7 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 8 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 9 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
# # python3 ../../expertnet.py --dataset ards --n_clusters 10 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt


# # python3 ../expertnet.py --dataset sepsis --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention False >> Results/ExpertNet_Ablations.txt
# # python3 ../expertnet.py --dataset sepsis --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention False >> Results/ExpertNet_Ablations.txt
# # python3 ../expertnet.py --dataset sepsis --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention False >> Results/ExpertNet_Ablations.txt
# # python3 ../expertnet.py --dataset sepsis --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention False >> Results/ExpertNet_Ablations.txt

########################################################
########################################################
#################### SEPSIS dataset ####################
########################################################
########################################################

# echo "ExpertNet Ablations" >> Results/ExpertNet_Ablations.txt

# echo "Attention Train, Attention Test: F,T" >> Results/ExpertNet_Ablations.txt

# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 1 >> Results/ExpertNet_Ablations.txt

# echo "Attention Train, Attention Test: F,F" >> Results/ExpertNet_Ablations.txt

# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 0 >> Results/ExpertNet_Ablations.txt

# echo "Attention Train, Attention Test: T,F" >> Results/ExpertNet_Ablations.txt

# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt
# python3 ../../expertnet.py --dataset sepsis24 --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 10 >> Results/ExpertNet_Ablations.txt

echo "Attention Train, Attention Test: T,T" >> Results/ExpertNet_Ablations.txt

python3 ../../expertnet.py --dataset sepsis24 --n_clusters 2 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 3 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 4 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
python3 ../../expertnet.py --dataset sepsis24 --n_clusters 5 --alpha 1 --beta 10 --gamma 5 --delta 0.1 --log_interval 2 --n_runs 5 --n_epochs 50 --n_z 32 --attention 11 >> Results/ExpertNet_Ablations.txt
